/**
 * @file apbuart.c
 * @brief APBUART driver for GR740 — interrupt-driven TX, polled/interrupt RX.
 *
 * Implements circular TX/RX buffers (256B TX, 512B RX) with
 * interrupt-driven transmission and interrupt-driven reception.
 *
 * @reference GR740 User Manual, GRLIB APBUART IP Core
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "apbuart.h"
#include "../../config/hw_config.h"
#include "../../config/mission_config.h"
#include "../../bsp/irq_handler.h"
#include "../../bsp/gr740_init.h"

#include <stdint.h>
#include <stddef.h>

/* ======================================================================
 * Circular Buffer
 * ====================================================================== */

typedef struct {
    volatile uint8_t  buf[512];
    volatile uint32_t head;     /**< Write index */
    volatile uint32_t tail;     /**< Read index  */
    uint32_t          size;     /**< Buffer size (power of 2) */
} circ_buf_t;

/* ======================================================================
 * Private Data
 * ====================================================================== */

static circ_buf_t s_tx_buf;
static circ_buf_t s_rx_buf;
static uint32_t   s_uart_base = 0U;
static uint32_t   s_uart_irq = 0U;
static volatile uint8_t s_uart_initialized = 0U;

/* ======================================================================
 * Buffer Helpers
 * ====================================================================== */

static inline uint32_t buf_count(const circ_buf_t *b)
{
    return (b->head - b->tail) & (b->size - 1U);
}

static inline uint32_t buf_free(const circ_buf_t *b)
{
    return b->size - 1U - buf_count(b);
}

static inline int32_t buf_push(circ_buf_t *b, uint8_t val)
{
    uint32_t next = (b->head + 1U) & (b->size - 1U);
    if (next == b->tail) {
        return -1; /* Full */
    }
    b->buf[b->head] = val;
    b->head = next;
    return 0;
}

static inline int32_t buf_pop(circ_buf_t *b, uint8_t *val)
{
    if (b->head == b->tail) {
        return -1; /* Empty */
    }
    *val = b->buf[b->tail];
    b->tail = (b->tail + 1U) & (b->size - 1U);
    return 0;
}

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Initialize UART with interrupt-driven TX and RX.
 * @param[in] base_addr UART base address.
 * @param[in] baudrate  Baud rate.
 * @param[in] irq_num   IRQ number.
 * @return UART_OK on success.
 */
int32_t uart_init(uint32_t base_addr, uint32_t baudrate, uint32_t irq_num)
{
    uint32_t scaler_val;
    int32_t  rc;

    if ((base_addr == 0U) || (baudrate == 0U)) {
        return UART_ERR_PARAM;
    }

    s_uart_base = base_addr;
    s_uart_irq = irq_num;

    /* Initialize circular buffers */
    s_tx_buf.head = 0U;
    s_tx_buf.tail = 0U;
    s_tx_buf.size = UART_TX_BUF_SIZE;

    s_rx_buf.head = 0U;
    s_rx_buf.tail = 0U;
    s_rx_buf.size = UART_RX_BUF_SIZE;

    /* Disable UART */
    REG_WRITE(s_uart_base, UART_CTRL_REG, 0U);

    /* Set baud rate scaler */
    scaler_val = (SYS_CLK_HZ / (baudrate * 8U + 7U)) - 1U;
    REG_WRITE(s_uart_base, UART_SCALER_REG, scaler_val);

    /* Clear status */
    (void)REG_READ(s_uart_base, UART_STATUS_REG);

    /* Register ISR */
    rc = irq_register(irq_num, uart_isr);
    if (rc != 0) {
        return UART_ERR_INIT;
    }
    (void)irq_enable(irq_num);

    /* Enable TX, RX, and RX interrupt */
    REG_WRITE(s_uart_base, UART_CTRL_REG,
              UART_CTRL_TE | UART_CTRL_RE | UART_CTRL_RI);

    s_uart_initialized = 1U;

    return UART_OK;
}

/**
 * @brief Send a single character.
 * @param[in] c Character.
 * @return UART_OK on success.
 */
int32_t uart_putc(char c)
{
    int32_t rc;

    if (s_uart_initialized == 0U) {
        return UART_ERR_INIT;
    }

    /* Push to TX buffer */
    uint32_t istate = bsp_disable_interrupts();
    rc = buf_push(&s_tx_buf, (uint8_t)c);
    bsp_restore_interrupts(istate);

    if (rc != 0) {
        return UART_ERR_TIMEOUT; /* Buffer full */
    }

    /* Enable TX interrupt to start sending */
    REG_SET(s_uart_base, UART_CTRL_REG, UART_CTRL_TI);

    return UART_OK;
}

/**
 * @brief Send a null-terminated string.
 * @param[in] str String.
 * @return UART_OK on success.
 */
int32_t uart_puts(const char *str)
{
    if (str == NULL) {
        return UART_ERR_PARAM;
    }
    if (s_uart_initialized == 0U) {
        return UART_ERR_INIT;
    }

    while (*str != '\0') {
        int32_t rc = uart_putc(*str);
        if (rc != UART_OK) {
            return rc;
        }
        str++;
    }

    return UART_OK;
}

/**
 * @brief Get a single character with timeout.
 * @param[out] c          Received character.
 * @param[in]  timeout_ms Timeout.
 * @return UART_OK on success.
 */
int32_t uart_getc(char *c, uint32_t timeout_ms)
{
    uint32_t start;
    uint8_t  val;

    if (c == NULL) {
        return UART_ERR_PARAM;
    }
    if (s_uart_initialized == 0U) {
        return UART_ERR_INIT;
    }

    start = bsp_get_uptime_ms();

    while (1) {
        uint32_t istate = bsp_disable_interrupts();
        int32_t rc = buf_pop(&s_rx_buf, &val);
        bsp_restore_interrupts(istate);

        if (rc == 0) {
            *c = (char)val;
            return UART_OK;
        }

        uint32_t elapsed = bsp_get_uptime_ms() - start;
        if (elapsed >= timeout_ms) {
            return UART_ERR_TIMEOUT;
        }
    }
}

/**
 * @brief Read a line (up to newline or max_len).
 * @param[out] buf        Buffer.
 * @param[in]  max_len    Max chars.
 * @param[in]  timeout_ms Timeout.
 * @return Number of characters read, or negative on error.
 */
int32_t uart_read_line(char *buf, uint32_t max_len, uint32_t timeout_ms)
{
    uint32_t count = 0U;
    char c;
    int32_t rc;

    if ((buf == NULL) || (max_len == 0U)) {
        return UART_ERR_PARAM;
    }

    while (count < (max_len - 1U)) {
        rc = uart_getc(&c, timeout_ms);
        if (rc != UART_OK) {
            break;
        }

        buf[count] = c;
        count++;

        if ((c == '\n') || (c == '\r')) {
            break;
        }
    }

    buf[count] = '\0';
    return (int32_t)count;
}

/**
 * @brief UART interrupt service routine.
 */
void uart_isr(void)
{
    uint32_t status = REG_READ(s_uart_base, UART_STATUS_REG);

    /* RX interrupt — data available */
    if ((status & UART_STATUS_DR) != 0U) {
        uint32_t data = REG_READ(s_uart_base, UART_DATA_REG);
        (void)buf_push(&s_rx_buf, (uint8_t)(data & 0xFFU));
    }

    /* TX interrupt — FIFO has space */
    if ((status & UART_STATUS_TE) != 0U) {
        uint8_t val;
        if (buf_pop(&s_tx_buf, &val) == 0) {
            REG_WRITE(s_uart_base, UART_DATA_REG, (uint32_t)val);
        } else {
            /* No more data to send — disable TX interrupt */
            REG_CLR(s_uart_base, UART_CTRL_REG, UART_CTRL_TI);
        }
    }

    /* Overrun — log error */
    if ((status & UART_STATUS_OV) != 0U) {
        /* Clear overrun by reading data register */
        (void)REG_READ(s_uart_base, UART_DATA_REG);
    }
}

/**
 * @brief Send binary data.
 * @param[in] data Data buffer.
 * @param[in] len  Length.
 * @return UART_OK on success.
 */
int32_t uart_write(const uint8_t *data, uint32_t len)
{
    uint32_t i;

    if ((data == NULL) && (len > 0U)) {
        return UART_ERR_PARAM;
    }

    for (i = 0U; i < len; i++) {
        int32_t rc = uart_putc((char)data[i]);
        if (rc != UART_OK) {
            return rc;
        }
    }

    return UART_OK;
}

/**
 * @brief Read binary data.
 * @param[out] data       Buffer.
 * @param[in]  len        Max bytes.
 * @param[in]  timeout_ms Timeout.
 * @return Bytes read (>=0), negative on error.
 */
int32_t uart_read(uint8_t *data, uint32_t len, uint32_t timeout_ms)
{
    uint32_t count = 0U;
    char c;

    if ((data == NULL) || (len == 0U)) {
        return UART_ERR_PARAM;
    }

    while (count < len) {
        int32_t rc = uart_getc(&c, timeout_ms);
        if (rc != UART_OK) {
            break;
        }
        data[count] = (uint8_t)c;
        count++;
    }

    return (int32_t)count;
}
