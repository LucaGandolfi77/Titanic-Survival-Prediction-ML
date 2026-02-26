/**
 * @file i2cmst.c
 * @brief I2CMST I2C master driver for GR740.
 *
 * Register-level implementation for the Cobham Gaisler I2CMST core.
 * Based on OpenCores I2C controller IP with Gaisler wrapper.
 * Supports 7-bit addressing, standard and fast mode.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "i2cmst.h"
#include "../../config/hw_config.h"

/* ── I2CMST Register Offsets ───────────────────────────────────────────── */
#define I2C_PRER_LO     0x00U   /**< Clock prescale low byte   */
#define I2C_PRER_HI     0x04U   /**< Clock prescale high byte  */
#define I2C_CTR         0x08U   /**< Control register          */
#define I2C_TXR         0x0CU   /**< Transmit register (W)     */
#define I2C_RXR         0x0CU   /**< Receive register (R)      */
#define I2C_CR          0x10U   /**< Command register (W)      */
#define I2C_SR          0x10U   /**< Status register (R)       */

/* ── Control Register Bits ─────────────────────────────────────────────── */
#define CTR_EN          (1U << 7)   /**< Core enable           */
#define CTR_IEN         (1U << 6)   /**< Interrupt enable      */

/* ── Command Register Bits ─────────────────────────────────────────────── */
#define CR_STA          (1U << 7)   /**< Generate START        */
#define CR_STO          (1U << 6)   /**< Generate STOP         */
#define CR_RD           (1U << 5)   /**< Read from slave       */
#define CR_WR           (1U << 4)   /**< Write to slave        */
#define CR_ACK          (1U << 3)   /**< ACK (0) / NACK (1)    */
#define CR_IACK         (1U << 0)   /**< Interrupt acknowledge */

/* ── Status Register Bits ──────────────────────────────────────────────── */
#define SR_RXACK        (1U << 7)   /**< Received ACK (0=ACK)  */
#define SR_BUSY         (1U << 6)   /**< I2C bus busy          */
#define SR_AL           (1U << 5)   /**< Arbitration lost      */
#define SR_TIP          (1U << 1)   /**< Transfer in progress  */
#define SR_IF           (1U << 0)   /**< Interrupt flag        */

/* ── Timeout ───────────────────────────────────────────────────────────── */
#define I2C_POLL_TIMEOUT  200000U   /**< Polling iterations    */

/* ── Module state ──────────────────────────────────────────────────────── */
static volatile uint32_t *i2c_base = (volatile uint32_t *)0;
static uint8_t i2c_initialized = 0U;

/* ── Register access ───────────────────────────────────────────────────── */
static inline uint32_t i2c_reg_read(uint32_t offset)
{
    return *(volatile uint32_t *)((uint32_t)i2c_base + offset);
}

static inline void i2c_reg_write(uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uint32_t)i2c_base + offset) = value;
}

/**
 * @brief Wait for transfer to complete (TIP → 0).
 * @return I2C_OK or I2C_ERR_TIMEOUT / I2C_ERR_ARB_LOST.
 */
static int32_t i2c_wait_tip(void)
{
    uint32_t sr;
    uint32_t timeout = I2C_POLL_TIMEOUT;

    do {
        sr = i2c_reg_read(I2C_SR);
        timeout--;
    } while (((sr & SR_TIP) != 0U) && (timeout > 0U));

    if (timeout == 0U) {
        return I2C_ERR_TIMEOUT;
    }

    if ((sr & SR_AL) != 0U) {
        return I2C_ERR_ARB_LOST;
    }

    return I2C_OK;
}

/**
 * @brief Check RXACK after transmit. ACK = bit 7 clear.
 * @return I2C_OK or I2C_ERR_NACK.
 */
static int32_t i2c_check_ack(void)
{
    uint32_t sr = i2c_reg_read(I2C_SR);
    if ((sr & SR_RXACK) != 0U) {
        /* NACK received — generate STOP */
        i2c_reg_write(I2C_CR, CR_STO);
        return I2C_ERR_NACK;
    }
    return I2C_OK;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t i2c_init(uint32_t base_addr, uint32_t clock_hz)
{
    uint32_t prescale;
    uint32_t sys_clk = 50000000U; /* 50 MHz */

    if (base_addr == 0U) {
        return I2C_ERR_PARAM;
    }
    if ((clock_hz != 100000U) && (clock_hz != 400000U)) {
        return I2C_ERR_PARAM;
    }

    i2c_base = (volatile uint32_t *)(uintptr_t)base_addr;

    /* Disable core during configuration */
    i2c_reg_write(I2C_CTR, 0U);

    /*
     * Prescale formula (OpenCores I2C):
     *   prescale = (sys_clk / (5 * SCL)) - 1
     *
     * 100 kHz: (50e6 / 5e5) - 1 = 99
     * 400 kHz: (50e6 / 2e6) - 1 = 24
     */
    prescale = (sys_clk / (5U * clock_hz)) - 1U;

    i2c_reg_write(I2C_PRER_LO, prescale & 0xFFU);
    i2c_reg_write(I2C_PRER_HI, (prescale >> 8U) & 0xFFU);

    /* Enable core, no interrupts (polled mode) */
    i2c_reg_write(I2C_CTR, CTR_EN);

    i2c_initialized = 1U;
    return I2C_OK;
}

int32_t i2c_write(uint8_t slave_addr, const uint8_t *data, uint32_t len)
{
    uint32_t i;
    int32_t  ret;
    uint8_t  addr_byte;

    if (i2c_initialized == 0U) {
        return I2C_ERR_INIT;
    }
    if (data == (const uint8_t *)0) {
        return I2C_ERR_PARAM;
    }
    if (len == 0U) {
        return I2C_ERR_PARAM;
    }

    /* Send slave address with W bit (bit 0 = 0) */
    addr_byte = (uint8_t)((slave_addr << 1U) & 0xFEU);
    i2c_reg_write(I2C_TXR, (uint32_t)addr_byte);
    i2c_reg_write(I2C_CR, CR_STA | CR_WR);

    ret = i2c_wait_tip();
    if (ret != I2C_OK) {
        return ret;
    }

    ret = i2c_check_ack();
    if (ret != I2C_OK) {
        return ret;
    }

    /* Send data bytes */
    for (i = 0U; i < len; i++) {
        i2c_reg_write(I2C_TXR, (uint32_t)data[i]);

        if (i == (len - 1U)) {
            /* Last byte: generate STOP */
            i2c_reg_write(I2C_CR, CR_WR | CR_STO);
        } else {
            i2c_reg_write(I2C_CR, CR_WR);
        }

        ret = i2c_wait_tip();
        if (ret != I2C_OK) {
            return ret;
        }

        ret = i2c_check_ack();
        if (ret != I2C_OK) {
            return ret;
        }
    }

    return I2C_OK;
}

int32_t i2c_read(uint8_t slave_addr, uint8_t *data, uint32_t len)
{
    uint32_t i;
    int32_t  ret;
    uint8_t  addr_byte;

    if (i2c_initialized == 0U) {
        return I2C_ERR_INIT;
    }
    if (data == (uint8_t *)0) {
        return I2C_ERR_PARAM;
    }
    if (len == 0U) {
        return I2C_ERR_PARAM;
    }

    /* Send slave address with R bit (bit 0 = 1) */
    addr_byte = (uint8_t)((slave_addr << 1U) | 0x01U);
    i2c_reg_write(I2C_TXR, (uint32_t)addr_byte);
    i2c_reg_write(I2C_CR, CR_STA | CR_WR);

    ret = i2c_wait_tip();
    if (ret != I2C_OK) {
        return ret;
    }

    ret = i2c_check_ack();
    if (ret != I2C_OK) {
        return ret;
    }

    /* Read data bytes */
    for (i = 0U; i < len; i++) {
        if (i == (len - 1U)) {
            /* Last byte: NACK + STOP */
            i2c_reg_write(I2C_CR, CR_RD | CR_ACK | CR_STO);
        } else {
            /* ACK (ACK bit = 0 means send ACK) */
            i2c_reg_write(I2C_CR, CR_RD);
        }

        ret = i2c_wait_tip();
        if (ret != I2C_OK) {
            return ret;
        }

        data[i] = (uint8_t)(i2c_reg_read(I2C_RXR) & 0xFFU);
    }

    return I2C_OK;
}

int32_t i2c_write_read(uint8_t slave_addr,
                        const uint8_t *tx_data, uint32_t tx_len,
                        uint8_t *rx_data, uint32_t rx_len)
{
    uint32_t i;
    int32_t  ret;
    uint8_t  addr_byte;

    if (i2c_initialized == 0U) {
        return I2C_ERR_INIT;
    }
    if ((tx_data == (const uint8_t *)0) || (rx_data == (uint8_t *)0)) {
        return I2C_ERR_PARAM;
    }
    if ((tx_len == 0U) || (rx_len == 0U)) {
        return I2C_ERR_PARAM;
    }

    /* --- Write phase --- */
    addr_byte = (uint8_t)((slave_addr << 1U) & 0xFEU);
    i2c_reg_write(I2C_TXR, (uint32_t)addr_byte);
    i2c_reg_write(I2C_CR, CR_STA | CR_WR);

    ret = i2c_wait_tip();
    if (ret != I2C_OK) { return ret; }
    ret = i2c_check_ack();
    if (ret != I2C_OK) { return ret; }

    for (i = 0U; i < tx_len; i++) {
        i2c_reg_write(I2C_TXR, (uint32_t)tx_data[i]);
        i2c_reg_write(I2C_CR, CR_WR);

        ret = i2c_wait_tip();
        if (ret != I2C_OK) { return ret; }
        ret = i2c_check_ack();
        if (ret != I2C_OK) { return ret; }
    }

    /* --- Repeated START + Read phase --- */
    addr_byte = (uint8_t)((slave_addr << 1U) | 0x01U);
    i2c_reg_write(I2C_TXR, (uint32_t)addr_byte);
    i2c_reg_write(I2C_CR, CR_STA | CR_WR);

    ret = i2c_wait_tip();
    if (ret != I2C_OK) { return ret; }
    ret = i2c_check_ack();
    if (ret != I2C_OK) { return ret; }

    for (i = 0U; i < rx_len; i++) {
        if (i == (rx_len - 1U)) {
            i2c_reg_write(I2C_CR, CR_RD | CR_ACK | CR_STO);
        } else {
            i2c_reg_write(I2C_CR, CR_RD);
        }

        ret = i2c_wait_tip();
        if (ret != I2C_OK) { return ret; }

        rx_data[i] = (uint8_t)(i2c_reg_read(I2C_RXR) & 0xFFU);
    }

    return I2C_OK;
}

int32_t i2c_read_reg(uint8_t slave_addr, uint8_t reg_addr, uint8_t *value)
{
    if (value == (uint8_t *)0) {
        return I2C_ERR_PARAM;
    }
    return i2c_write_read(slave_addr, &reg_addr, 1U, value, 1U);
}

int32_t i2c_write_reg(uint8_t slave_addr, uint8_t reg_addr, uint8_t value)
{
    uint8_t buf[2];
    buf[0] = reg_addr;
    buf[1] = value;
    return i2c_write(slave_addr, buf, 2U);
}
