/**
 * @file grspw.c
 * @brief GRSPW2 SpaceWire driver for GR740 (4 links).
 *
 * Register-level driver with DMA descriptor rings, auto-start,
 * auto-recovery, and configurable link speeds.
 *
 * @reference GR740 User Manual, GRLIB GRSPW2 IP Core, ECSS-E-ST-50-12C
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "grspw.h"
#include "../../config/hw_config.h"
#include "../../config/mission_config.h"
#include "../../bsp/irq_handler.h"
#include "../../bsp/gr740_init.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ======================================================================
 * DMA Descriptor Format
 * GRSPW2 TX descriptor (4 words):
 *   Word 0: Control (header len, data len, flags)
 *   Word 1: Header pointer
 *   Word 2: Data pointer
 *   Word 3: Reserved / next descriptor pointer
 *
 * GRSPW2 RX descriptor (4 words):
 *   Word 0: Control (max len, flags)
 *   Word 1: Data pointer
 *   Word 2: Reserved
 *   Word 3: Reserved
 * ====================================================================== */

/** SpW DMA descriptor */
typedef struct {
    volatile uint32_t ctrl;     /**< Control word                        */
    volatile uint32_t addr;     /**< Header/Data pointer                 */
    volatile uint32_t data;     /**< Data pointer (TX) / reserved (RX)   */
    volatile uint32_t next;     /**< Reserved / next pointer             */
} spw_dma_desc_t;

/* TX descriptor control bits */
#define SPW_TX_DESC_EN          (1U << 12)  /**< Descriptor enable       */
#define SPW_TX_DESC_WR          (1U << 13)  /**< Wrap (last in ring)     */
#define SPW_TX_DESC_IE          (1U << 14)  /**< Interrupt enable        */
#define SPW_TX_DESC_LE          (1U << 15)  /**< Link error              */
#define SPW_TX_DESC_HC_SHIFT    0           /**< Header CRC              */
#define SPW_TX_DESC_DC_SHIFT    16          /**< Data CRC                */

/* RX descriptor control bits */
#define SPW_RX_DESC_EN          (1U << 25)  /**< Descriptor enable       */
#define SPW_RX_DESC_WR          (1U << 26)  /**< Wrap (last in ring)     */
#define SPW_RX_DESC_IE          (1U << 27)  /**< Interrupt enable        */
#define SPW_RX_DESC_TR          (1U << 28)  /**< Truncated               */
#define SPW_RX_DESC_DC          (1U << 29)  /**< Data CRC error          */
#define SPW_RX_DESC_HC          (1U << 30)  /**< Header CRC error        */
#define SPW_RX_DESC_EEP         (1U << 31)  /**< EEP termination         */
#define SPW_RX_DESC_LEN_MASK    0x01FFFFFFU /**< Received length         */

/* ======================================================================
 * Per-Port State
 * ====================================================================== */

/** TX/RX data buffers — one per descriptor per port */
static uint8_t s_tx_bufs[SPW_PORT_COUNT][SPW_TX_DESC_COUNT][SPW_MAX_PACKET_SIZE]
    __attribute__((aligned(4)));
static uint8_t s_rx_bufs[SPW_PORT_COUNT][SPW_RX_DESC_COUNT][SPW_MAX_PACKET_SIZE]
    __attribute__((aligned(4)));

/** DMA descriptor rings per port */
static spw_dma_desc_t s_tx_descs[SPW_PORT_COUNT][SPW_TX_DESC_COUNT]
    __attribute__((aligned(1024)));
static spw_dma_desc_t s_rx_descs[SPW_PORT_COUNT][SPW_RX_DESC_COUNT]
    __attribute__((aligned(1024)));

/** Per-port state */
typedef struct {
    uint32_t base_addr;         /**< Register base address               */
    uint32_t irq_num;           /**< IRQ number                          */
    volatile uint32_t tx_head;  /**< TX ring head (next to submit)       */
    volatile uint32_t rx_head;  /**< RX ring head (next to check)        */
    volatile uint8_t  initialized;
    spw_status_t status;        /**< Status counters                     */
} spw_port_state_t;

static spw_port_state_t s_ports[SPW_PORT_COUNT];

/* Port base address lookup */
static const uint32_t s_port_bases[SPW_PORT_COUNT] = {
    GRSPW0_BASE, GRSPW1_BASE, GRSPW2_BASE, GRSPW3_BASE
};
static const uint32_t s_port_irqs[SPW_PORT_COUNT] = {
    IRQ_GRSPW0, IRQ_GRSPW1, IRQ_GRSPW2, IRQ_GRSPW3
};

/* ======================================================================
 * ISR Wrappers (one per port, dispatches to common handler)
 * ====================================================================== */
static void spw_isr_port0(void) { spw_isr(0); }
static void spw_isr_port1(void) { spw_isr(1); }
static void spw_isr_port2(void) { spw_isr(2); }
static void spw_isr_port3(void) { spw_isr(3); }

static const irq_handler_fn s_isr_table[SPW_PORT_COUNT] = {
    spw_isr_port0, spw_isr_port1, spw_isr_port2, spw_isr_port3
};

/* ======================================================================
 * Forward Declarations
 * ====================================================================== */

static int32_t spw_setup_dma(uint8_t port);
static uint32_t spw_calc_clkdiv(uint32_t link_speed_mbps);
static void spw_setup_rx_desc(uint8_t port, uint32_t idx);

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Initialize a SpaceWire port.
 * @param[in] port           Port number (0–3).
 * @param[in] link_speed_mbps Link speed (10, 50, 100, 200).
 * @return SPW_OK on success.
 */
int32_t spw_init(uint8_t port, uint32_t link_speed_mbps)
{
    uint32_t base;
    uint32_t clkdiv;
    int32_t  rc;

    if (port >= SPW_PORT_COUNT) {
        return SPW_ERR_PARAM;
    }
    if ((link_speed_mbps == 0U) || (link_speed_mbps > 200U)) {
        return SPW_ERR_PARAM;
    }

    base = s_port_bases[port];
    s_ports[port].base_addr = base;
    s_ports[port].irq_num = s_port_irqs[port];
    s_ports[port].tx_head = 0U;
    s_ports[port].rx_head = 0U;
    (void)memset(&s_ports[port].status, 0, sizeof(spw_status_t));

    /* Step 1: Disable and reset the link */
    REG_WRITE(base, GRSPW_CTRL, GRSPW_CTRL_LD);

    /* Step 2: Clear status register (write-1-to-clear) */
    REG_WRITE(base, GRSPW_STATUS, 0xFFFFFFFFU);

    /* Step 3: Set node address (use port number as address) */
    REG_WRITE(base, GRSPW_NODEADDR, (uint32_t)(port + 1U));

    /* Step 4: Configure clock divisor for link speed */
    clkdiv = spw_calc_clkdiv(link_speed_mbps);
    REG_WRITE(base, GRSPW_CLKDIV, clkdiv);

    /* Step 5: Setup DMA descriptor rings */
    rc = spw_setup_dma(port);
    if (rc != SPW_OK) {
        return rc;
    }

    /* Step 6: Register ISR */
    rc = irq_register(s_ports[port].irq_num, s_isr_table[port]);
    if (rc != 0) {
        return SPW_ERR_INIT;
    }
    (void)irq_enable(s_ports[port].irq_num);

    /* Step 7: Enable DMA channel */
    REG_WRITE(base, GRSPW_DMA0_CTRL,
              GRSPW_DMA_CTRL_TE | GRSPW_DMA_CTRL_RE |
              GRSPW_DMA_CTRL_TI | GRSPW_DMA_CTRL_RI |
              GRSPW_DMA_CTRL_NS);

    /* Step 8: Set max RX packet length */
    REG_WRITE(base, GRSPW_DMA0_RXMAX, SPW_MAX_PACKET_SIZE);

    /* Step 9: Start the link (auto-start, RMAP enable, interrupts) */
    REG_WRITE(base, GRSPW_CTRL,
              GRSPW_CTRL_LS | GRSPW_CTRL_AS | GRSPW_CTRL_IE |
              GRSPW_CTRL_RE | GRSPW_CTRL_TE | GRSPW_CTRL_TR);

    s_ports[port].initialized = 1U;

    return SPW_OK;
}

/**
 * @brief Send a packet over SpaceWire.
 * @param[in] port Port number (0–3).
 * @param[in] data Packet data.
 * @param[in] len  Packet length.
 * @return SPW_OK on success.
 */
int32_t spw_send(uint8_t port, const uint8_t *data, uint32_t len)
{
    uint32_t base;
    uint32_t head;
    spw_dma_desc_t *desc;

    if (port >= SPW_PORT_COUNT) {
        return SPW_ERR_PARAM;
    }
    if ((data == NULL) || (len == 0U)) {
        return SPW_ERR_PARAM;
    }
    if (len > SPW_MAX_PACKET_SIZE) {
        return SPW_ERR_PARAM;
    }
    if (s_ports[port].initialized == 0U) {
        return SPW_ERR_INIT;
    }

    base = s_ports[port].base_addr;
    head = s_ports[port].tx_head;

    /* Check if descriptor is available (EN bit must be 0 = completed) */
    desc = &s_tx_descs[port][head];
    if ((desc->ctrl & SPW_TX_DESC_EN) != 0U) {
        return SPW_ERR_FULL;
    }

    /* Copy data to TX buffer */
    (void)memcpy(s_tx_bufs[port][head], data, len);

    /* Setup descriptor */
    desc->data = (uint32_t)s_tx_bufs[port][head];
    desc->addr = 0U; /* No header pointer for raw data */

    /* Control: enable, data length in lower 24 bits, IE for interrupt */
    uint32_t ctrl = SPW_TX_DESC_EN | SPW_TX_DESC_IE | (len & 0x00FFFFFFU);
    if (head == (SPW_TX_DESC_COUNT - 1U)) {
        ctrl |= SPW_TX_DESC_WR; /* Wrap at end of ring */
    }
    desc->ctrl = ctrl;

    /* Advance head */
    s_ports[port].tx_head = (head + 1U) % SPW_TX_DESC_COUNT;
    s_ports[port].status.tx_packets++;

    /* Kick DMA — write TX descriptor table pointer */
    REG_WRITE(base, GRSPW_DMA0_TXDESC, (uint32_t)s_tx_descs[port]);

    return SPW_OK;
}

/**
 * @brief Receive a packet from SpaceWire with timeout.
 * @param[in]  port       Port number (0–3).
 * @param[out] buf        Receive buffer.
 * @param[in,out] len     Buffer size (in) / received length (out).
 * @param[in]  timeout_ms Timeout.
 * @return SPW_OK on success, SPW_ERR_TIMEOUT on timeout.
 */
int32_t spw_receive(uint8_t port, uint8_t *buf, uint32_t *len,
                    uint32_t timeout_ms)
{
    uint32_t start_ms;
    uint32_t head;
    spw_dma_desc_t *desc;
    uint32_t rx_len;
    uint32_t buf_size;

    if (port >= SPW_PORT_COUNT) {
        return SPW_ERR_PARAM;
    }
    if ((buf == NULL) || (len == NULL)) {
        return SPW_ERR_PARAM;
    }
    if (s_ports[port].initialized == 0U) {
        return SPW_ERR_INIT;
    }

    buf_size = *len;
    start_ms = bsp_get_uptime_ms();

    while (1) {
        head = s_ports[port].rx_head;
        desc = &s_rx_descs[port][head];

        /* Check if descriptor has been completed (EN bit cleared by HW) */
        if ((desc->ctrl & SPW_RX_DESC_EN) == 0U) {
            /* Check for errors */
            if ((desc->ctrl & SPW_RX_DESC_TR) != 0U) {
                /* Truncated packet */
                spw_setup_rx_desc(port, head);
                s_ports[port].rx_head = (head + 1U) % SPW_RX_DESC_COUNT;
                continue;
            }
            if ((desc->ctrl & SPW_RX_DESC_EEP) != 0U) {
                /* EEP termination — link error */
                spw_setup_rx_desc(port, head);
                s_ports[port].rx_head = (head + 1U) % SPW_RX_DESC_COUNT;
                s_ports[port].status.disconnect_errors++;
                continue;
            }

            /* Get received length */
            rx_len = desc->ctrl & SPW_RX_DESC_LEN_MASK;
            if (rx_len > buf_size) {
                rx_len = buf_size;
            }

            /* Copy data out */
            (void)memcpy(buf, s_rx_bufs[port][head], rx_len);
            *len = rx_len;

            /* Re-arm descriptor */
            spw_setup_rx_desc(port, head);
            s_ports[port].rx_head = (head + 1U) % SPW_RX_DESC_COUNT;
            s_ports[port].status.rx_packets++;

            return SPW_OK;
        }

        /* Timeout check */
        uint32_t elapsed = bsp_get_uptime_ms() - start_ms;
        if (elapsed >= timeout_ms) {
            *len = 0U;
            return SPW_ERR_TIMEOUT;
        }
    }
}

/**
 * @brief Get SpaceWire link status.
 * @param[in]  port   Port (0–3).
 * @param[out] status Status structure.
 * @return SPW_OK on success.
 */
int32_t spw_get_status(uint8_t port, spw_status_t *status)
{
    uint32_t base;
    uint32_t stat_reg;

    if (port >= SPW_PORT_COUNT) {
        return SPW_ERR_PARAM;
    }
    if (status == NULL) {
        return SPW_ERR_PARAM;
    }

    base = s_ports[port].base_addr;
    stat_reg = REG_READ(base, GRSPW_STATUS);

    /* Extract link state from status register bits [23:21] */
    uint32_t ls = (stat_reg & GRSPW_STATUS_LS_MASK) >> 21;
    s_ports[port].status.state = (spw_link_state_t)ls;

    *status = s_ports[port].status;

    return SPW_OK;
}

/**
 * @brief Check if SpaceWire link is in RUN state.
 * @param[in] port Port (0–3).
 * @return 1 if running, 0 if not.
 */
int32_t spw_link_is_running(uint8_t port)
{
    uint32_t stat;

    if (port >= SPW_PORT_COUNT) {
        return SPW_ERR_PARAM;
    }

    stat = REG_READ(s_ports[port].base_addr, GRSPW_STATUS);
    uint32_t ls = (stat & GRSPW_STATUS_LS_MASK) >> 21;

    return (ls == (uint32_t)SPW_LS_RUN) ? 1 : 0;
}

/**
 * @brief SpaceWire ISR handler.
 * @param[in] port Port number.
 */
void spw_isr(uint8_t port)
{
    uint32_t base;
    uint32_t status;

    if (port >= SPW_PORT_COUNT) {
        return;
    }

    base = s_ports[port].base_addr;
    status = REG_READ(base, GRSPW_STATUS);

    /* Credit error */
    if ((status & GRSPW_STATUS_CE) != 0U) {
        s_ports[port].status.credit_errors++;
    }

    /* Disconnect error — attempt auto-recovery */
    if ((status & GRSPW_STATUS_DE) != 0U) {
        s_ports[port].status.disconnect_errors++;
        /* Re-enable link start for auto-recovery */
        REG_SET(base, GRSPW_CTRL, GRSPW_CTRL_LS | GRSPW_CTRL_AS);
    }

    /* Parity error */
    if ((status & GRSPW_STATUS_PE) != 0U) {
        s_ports[port].status.parity_errors++;
    }

    /* Escape error */
    if ((status & GRSPW_STATUS_ER) != 0U) {
        s_ports[port].status.escape_errors++;
    }

    /* Clear all status bits (write-1-to-clear) */
    REG_WRITE(base, GRSPW_STATUS, status);
}

/* ======================================================================
 * Private Functions
 * ====================================================================== */

/**
 * @brief Setup DMA descriptor rings for a port.
 * @param[in] port Port number.
 * @return SPW_OK on success.
 */
static int32_t spw_setup_dma(uint8_t port)
{
    uint32_t i;
    uint32_t base = s_ports[port].base_addr;

    /* Initialize TX descriptors */
    (void)memset(s_tx_descs[port], 0, sizeof(s_tx_descs[port]));
    for (i = 0U; i < SPW_TX_DESC_COUNT; i++) {
        s_tx_descs[port][i].ctrl = 0U; /* Not enabled */
        s_tx_descs[port][i].data = (uint32_t)s_tx_bufs[port][i];
        if (i == (SPW_TX_DESC_COUNT - 1U)) {
            s_tx_descs[port][i].ctrl |= SPW_TX_DESC_WR; /* Wrap */
        }
    }

    /* Initialize RX descriptors */
    (void)memset(s_rx_descs[port], 0, sizeof(s_rx_descs[port]));
    for (i = 0U; i < SPW_RX_DESC_COUNT; i++) {
        spw_setup_rx_desc(port, i);
    }

    /* Set descriptor table base addresses */
    REG_WRITE(base, GRSPW_DMA0_TXDESC, (uint32_t)s_tx_descs[port]);
    REG_WRITE(base, GRSPW_DMA0_RXDESC, (uint32_t)s_rx_descs[port]);

    return SPW_OK;
}

/**
 * @brief Calculate clock divisor for a given link speed.
 * @param[in] link_speed_mbps Link speed in Mbps.
 * @return Clock divisor register value.
 */
static uint32_t spw_calc_clkdiv(uint32_t link_speed_mbps)
{
    /*
     * GRSPW2 clock divisor:
     * Link frequency = SYS_CLK / (clkdiv_run + 1)
     * Startup frequency must be ≤ 10 Mbps per SpW standard
     *
     * For 50 MHz sys clock:
     *   100 Mbps: clkdiv_run = 0 (50 / 1 = 50 MHz TX clock → 100 Mbps DDR)
     *    50 Mbps: clkdiv_run = 0
     *    10 Mbps: clkdiv_run = 4 (50 / 5 = 10 MHz → 10 Mbps)
     *
     * Start clock divisor (must be ≤ 10 MHz):
     *   clkdiv_start = 4 (50 / 5 = 10 MHz)
     *
     * Register format: [15:8] = clkdiv_start, [7:0] = clkdiv_run
     */

    uint32_t clkdiv_start = (SYS_CLK_HZ / 10000000U) - 1U; /* ~10 Mbps startup */
    uint32_t clkdiv_run;

    if (link_speed_mbps >= 200U) {
        clkdiv_run = 0U;
    } else if (link_speed_mbps >= 100U) {
        clkdiv_run = 0U;
    } else if (link_speed_mbps >= 50U) {
        clkdiv_run = 0U;
    } else {
        clkdiv_run = (SYS_CLK_HZ / (link_speed_mbps * 1000000U)) - 1U;
    }

    return ((clkdiv_start & 0xFFU) << 8) | (clkdiv_run & 0xFFU);
}

/**
 * @brief Setup/re-arm a single RX descriptor.
 * @param[in] port Port number.
 * @param[in] idx  Descriptor index.
 */
static void spw_setup_rx_desc(uint8_t port, uint32_t idx)
{
    s_rx_descs[port][idx].addr = (uint32_t)s_rx_bufs[port][idx];
    s_rx_descs[port][idx].data = 0U;
    s_rx_descs[port][idx].next = 0U;

    uint32_t ctrl = SPW_RX_DESC_EN | SPW_RX_DESC_IE;
    if (idx == (SPW_RX_DESC_COUNT - 1U)) {
        ctrl |= SPW_RX_DESC_WR; /* Wrap */
    }
    s_rx_descs[port][idx].ctrl = ctrl;
}
