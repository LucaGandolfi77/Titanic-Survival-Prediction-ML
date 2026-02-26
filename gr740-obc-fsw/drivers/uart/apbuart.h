/**
 * @file apbuart.h
 * @brief APBUART driver interface for GR740.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef APBUART_H
#define APBUART_H

#include <stdint.h>

#define UART_OK             0
#define UART_ERR_PARAM      (-1)
#define UART_ERR_INIT       (-2)
#define UART_ERR_TIMEOUT    (-3)
#define UART_ERR_OVERRUN    (-4)

/**
 * @brief Initialize UART peripheral with interrupt-driven TX.
 * @param[in] base_addr UART base address.
 * @param[in] baudrate  Baud rate.
 * @param[in] irq_num   IRQ number.
 * @return UART_OK on success.
 */
int32_t uart_init(uint32_t base_addr, uint32_t baudrate, uint32_t irq_num);

/**
 * @brief Send a single character (interrupt-driven).
 * @param[in] c Character to send.
 * @return UART_OK on success, UART_ERR_TIMEOUT on buffer full.
 */
int32_t uart_putc(char c);

/**
 * @brief Send a null-terminated string.
 * @param[in] str String to send.
 * @return UART_OK on success.
 */
int32_t uart_puts(const char *str);

/**
 * @brief Get a single character (blocking with timeout).
 * @param[out] c          Pointer to store received character.
 * @param[in]  timeout_ms Timeout in milliseconds.
 * @return UART_OK on success, UART_ERR_TIMEOUT on timeout.
 */
int32_t uart_getc(char *c, uint32_t timeout_ms);

/**
 * @brief Read a line from UART (blocking).
 * @param[out] buf        Buffer for received line.
 * @param[in]  max_len    Maximum characters to read.
 * @param[in]  timeout_ms Timeout.
 * @return Number of characters read (>=0), or negative on error.
 */
int32_t uart_read_line(char *buf, uint32_t max_len, uint32_t timeout_ms);

/**
 * @brief UART interrupt service routine.
 */
void uart_isr(void);

/**
 * @brief Send binary data over UART.
 * @param[in] data Data buffer.
 * @param[in] len  Data length.
 * @return UART_OK on success.
 */
int32_t uart_write(const uint8_t *data, uint32_t len);

/**
 * @brief Read binary data from UART.
 * @param[out] data       Buffer for received data.
 * @param[in]  len        Max bytes to read.
 * @param[in]  timeout_ms Timeout.
 * @return Number of bytes read (>=0), or negative on error.
 */
int32_t uart_read(uint8_t *data, uint32_t len, uint32_t timeout_ms);

#endif /* APBUART_H */
