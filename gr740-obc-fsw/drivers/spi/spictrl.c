/**
 * @file spictrl.c
 * @brief SPICTRL SPI master driver for GR740.
 *
 * Register-level implementation for the Cobham Gaisler SPICTRL core.
 * Supports CPOL=0, CPHA=0 (SPI mode 0) with configurable clock divider.
 * Used for MRAM access, external sensors, and subsystem SPI buses.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "spictrl.h"
#include "../../config/hw_config.h"

/* ── SPICTRL Register Offsets ──────────────────────────────────────────── */
#define SPICTRL_CAP         0x00U   /**< Capability register (RO) */
#define SPICTRL_MODE        0x20U   /**< Mode register            */
#define SPICTRL_EVENT       0x24U   /**< Event register           */
#define SPICTRL_MASK        0x28U   /**< Mask register            */
#define SPICTRL_CMD         0x2CU   /**< Command register         */
#define SPICTRL_TX          0x30U   /**< Transmit register        */
#define SPICTRL_RX          0x34U   /**< Receive register         */
#define SPICTRL_SLVSEL      0x38U   /**< Slave select register    */
#define SPICTRL_ASLVSEL     0x3CU   /**< Auto slave select        */

/* ── Mode Register Bits ────────────────────────────────────────────────── */
#define MODE_LOOP           (1U << 30)  /**< Loopback mode         */
#define MODE_CPOL           (1U << 29)  /**< Clock polarity        */
#define MODE_CPHA           (1U << 28)  /**< Clock phase           */
#define MODE_DIV16          (1U << 27)  /**< Divide by 16 prescale */
#define MODE_REV            (1U << 26)  /**< Reverse (MSB first)   */
#define MODE_MS             (1U << 25)  /**< Master/Slave (1=master) */
#define MODE_EN             (1U << 24)  /**< Enable SPI            */
#define MODE_LEN_SHIFT      20U         /**< Word length shift     */
#define MODE_LEN_MASK       0x00F00000U /**< Word length mask      */
#define MODE_PM_SHIFT       16U         /**< Prescale modulus shift */
#define MODE_PM_MASK        0x000F0000U /**< Prescale modulus mask */
#define MODE_FACT           (1U << 13)  /**< PM factor             */
#define MODE_ASEL           (1U << 14)  /**< Auto slave select     */
#define MODE_CG_SHIFT       7U          /**< Clock gap shift       */
#define MODE_CG_MASK        0x00001F80U /**< Clock gap mask        */

/* ── Event Register Bits ───────────────────────────────────────────────── */
#define EVENT_LT            (1U << 14)  /**< Last character        */
#define EVENT_OV            (1U << 12)  /**< Overrun               */
#define EVENT_UN            (1U << 11)  /**< Underrun              */
#define EVENT_MME           (1U << 10)  /**< Multi-master error    */
#define EVENT_NE            (1U << 9)   /**< Not empty (RX ready)  */
#define EVENT_NF            (1U << 8)   /**< Not full (TX ready)   */

/* ── Command Register Bits ─────────────────────────────────────────────── */
#define CMD_LST             (1U << 22)  /**< Last character        */

/* ── Timeout for polling ───────────────────────────────────────────────── */
#define SPI_POLL_TIMEOUT    100000U     /**< Polling iterations    */

/* ── Module state ──────────────────────────────────────────────────────── */
static volatile uint32_t *spi_base = (volatile uint32_t *)0;
static uint8_t            spi_initialized = 0U;

/* ── Register access helpers ───────────────────────────────────────────── */
static inline uint32_t spi_reg_read(uint32_t offset)
{
    return *(volatile uint32_t *)((uint32_t)spi_base + offset);
}

static inline void spi_reg_write(uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uint32_t)spi_base + offset) = value;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t spi_init(uint32_t base_addr, uint32_t clock_hz)
{
    uint32_t mode_val;
    uint32_t pm;
    uint32_t sys_clk = 50000000U;  /* 50 MHz system clock */

    if (base_addr == 0U) {
        return SPI_ERR_PARAM;
    }
    if (clock_hz == 0U) {
        return SPI_ERR_PARAM;
    }

    spi_base = (volatile uint32_t *)(uintptr_t)base_addr;

    /*
     * SPICTRL clock equation:
     *   f_sck = f_sys / (4 * (PM + 1))   when FACT=0
     *   f_sck = f_sys / (2 * (PM + 1))   when FACT=1
     *
     * Using FACT=0:
     *   PM = (f_sys / (4 * f_sck)) - 1
     *   For 10 MHz: PM = 50e6/(4*10e6) - 1 = 0.25 → PM = 0 → 12.5 MHz
     *   For accuracy, use ceiling.
     *
     * Using FACT=1 (finer control):
     *   PM = (f_sys / (2 * f_sck)) - 1
     *   For 10 MHz: PM = 50e6/(2*10e6) - 1 = 1.5 → PM = 1 → 16.67 MHz → PM=2 → 8.33 MHz
     *
     * We target close to requested without exceeding:
     *   PM = ceil(f_sys / (4 * f_sck)) - 1   (FACT=0)
     */
    pm = (sys_clk + (4U * clock_hz) - 1U) / (4U * clock_hz);
    if (pm > 0U) {
        pm = pm - 1U;
    }
    if (pm > 0x0FU) {
        pm = 0x0FU; /* Max prescale modulus is 4 bits */
    }

    /* Build mode register:
     * - Master mode (MS=1)
     * - Enable (EN=1)
     * - 8-bit word length (LEN=7, field = bits_per_word - 1)
     * - REV=1 (MSB first)
     * - CPOL=0, CPHA=0 (SPI mode 0)
     * - No loopback
     */
    mode_val = 0U;
    mode_val |= MODE_MS;               /* Master mode             */
    mode_val |= MODE_EN;               /* Enable                  */
    mode_val |= MODE_REV;              /* MSB first               */
    mode_val |= (7U << MODE_LEN_SHIFT) & MODE_LEN_MASK;  /* 8 bits */
    mode_val |= (pm << MODE_PM_SHIFT) & MODE_PM_MASK;

    /* Clear all events before enabling */
    spi_reg_write(SPICTRL_EVENT, 0xFFFFFFFFU);

    /* Disable interrupts (polled mode) */
    spi_reg_write(SPICTRL_MASK, 0U);

    /* Deselect all slaves */
    spi_reg_write(SPICTRL_SLVSEL, 0xFFFFFFFFU);

    /* Program mode register */
    spi_reg_write(SPICTRL_MODE, mode_val);

    spi_initialized = 1U;
    return SPI_OK;
}

int32_t spi_select(uint8_t slave)
{
    uint32_t slvsel;

    if (spi_initialized == 0U) {
        return SPI_ERR_INIT;
    }
    if (slave > 31U) {
        return SPI_ERR_PARAM;
    }

    /* Active low: clear the bit for the selected slave */
    slvsel = 0xFFFFFFFFU;
    slvsel &= ~(1U << slave);
    spi_reg_write(SPICTRL_SLVSEL, slvsel);

    return SPI_OK;
}

int32_t spi_deselect(void)
{
    if (spi_initialized == 0U) {
        return SPI_ERR_INIT;
    }

    /* All bits set = all slaves deselected */
    spi_reg_write(SPICTRL_SLVSEL, 0xFFFFFFFFU);

    return SPI_OK;
}

int32_t spi_transfer_byte(uint8_t tx_byte, uint8_t *rx_byte)
{
    uint32_t event;
    uint32_t timeout;
    uint32_t rx_data;

    if (spi_initialized == 0U) {
        return SPI_ERR_INIT;
    }

    /* Wait for TX FIFO not full */
    timeout = SPI_POLL_TIMEOUT;
    do {
        event = spi_reg_read(SPICTRL_EVENT);
        timeout--;
    } while (((event & EVENT_NF) == 0U) && (timeout > 0U));

    if (timeout == 0U) {
        return SPI_ERR_TIMEOUT;
    }

    /* Write TX data */
    spi_reg_write(SPICTRL_TX, (uint32_t)tx_byte);

    /* Wait for RX data available (NE = not empty) */
    timeout = SPI_POLL_TIMEOUT;
    do {
        event = spi_reg_read(SPICTRL_EVENT);
        timeout--;
    } while (((event & EVENT_NE) == 0U) && (timeout > 0U));

    if (timeout == 0U) {
        return SPI_ERR_TIMEOUT;
    }

    /* Read RX data */
    rx_data = spi_reg_read(SPICTRL_RX);

    if (rx_byte != (uint8_t *)0) {
        *rx_byte = (uint8_t)(rx_data & 0xFFU);
    }

    /* Clear events */
    spi_reg_write(SPICTRL_EVENT, EVENT_NE | EVENT_NF);

    return SPI_OK;
}

int32_t spi_transfer(const uint8_t *tx_buf, uint8_t *rx_buf, uint32_t len)
{
    uint32_t i;
    uint8_t  tx_byte;
    uint8_t  rx_byte;
    int32_t  ret;

    if (spi_initialized == 0U) {
        return SPI_ERR_INIT;
    }
    if (len == 0U) {
        return SPI_OK;
    }

    for (i = 0U; i < len; i++) {
        /* Determine TX byte */
        if (tx_buf != (const uint8_t *)0) {
            tx_byte = tx_buf[i];
        } else {
            tx_byte = 0xFFU; /* Default fill for read-only */
        }

        ret = spi_transfer_byte(tx_byte, &rx_byte);
        if (ret != SPI_OK) {
            return ret;
        }

        /* Store RX byte if buffer provided */
        if (rx_buf != (uint8_t *)0) {
            rx_buf[i] = rx_byte;
        }
    }

    return SPI_OK;
}

int32_t spi_write(const uint8_t *data, uint32_t len)
{
    if (data == (const uint8_t *)0) {
        return SPI_ERR_PARAM;
    }
    return spi_transfer(data, (uint8_t *)0, len);
}

int32_t spi_read(uint8_t *data, uint32_t len)
{
    if (data == (uint8_t *)0) {
        return SPI_ERR_PARAM;
    }
    return spi_transfer((const uint8_t *)0, data, len);
}
