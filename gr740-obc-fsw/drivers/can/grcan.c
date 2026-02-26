/**
 * @file grcan.c
 * @brief GRCAN CAN 2.0B controller driver for GR740.
 *
 * Full register-level driver for the GRCAN core on the GR740 SoC.
 * Supports CAN 2.0B extended frames (29-bit ID), TX/RX ring buffers
 * with configurable power-of-2 sizes, and automatic bus-off recovery.
 *
 * @reference GR740 User Manual, GRLIB GRCAN IP Core documentation
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 *
 * MISRA-C:2012 Deviations:
 *   - Rule 11.4: Cast between pointer and integer for HW register access.
 *   - Rule 11.6: Cast from integer to pointer for memory-mapped I/O.
 */

#include "grcan.h"
#include "../../config/hw_config.h"
#include "../../config/mission_config.h"
#include "../../bsp/irq_handler.h"
#include "../../bsp/gr740_init.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ======================================================================
 * GRCAN Hardware Descriptor Format
 * Each descriptor is 4 words (16 bytes):
 *   Word 0: [31] = Extended, [28:0] = CAN ID
 *   Word 1: [3:0] = DLC, [4] = RTR, [31:16] = Timestamp
 *   Word 2: Data bytes [0-3]
 *   Word 3: Data bytes [4-7]
 * ====================================================================== */

/** GRCAN DMA descriptor */
typedef struct {
    uint32_t word0;     /**< ID field + extended flag                    */
    uint32_t word1;     /**< DLC, RTR, timestamp                        */
    uint32_t data_hi;   /**< Data bytes 0-3                             */
    uint32_t data_lo;   /**< Data bytes 4-7                             */
} grcan_desc_t;

/* Descriptor word0 bits */
#define DESC_W0_EXT_BIT         (1U << 31)
#define DESC_W0_ID_EXT_MASK     0x1FFFFFFFU
#define DESC_W0_ID_STD_SHIFT    18U
#define DESC_W0_ID_STD_MASK     (0x7FFU << 18)

/* Descriptor word1 bits */
#define DESC_W1_DLC_MASK        0x0FU
#define DESC_W1_RTR_BIT         (1U << 4)
#define DESC_W1_SYNC_BIT        (1U << 5)

/* ======================================================================
 * Private Data
 * ====================================================================== */

/** TX descriptor ring (aligned to 1KB boundary) */
static grcan_desc_t s_tx_ring[CAN_TX_RING_SIZE]
    __attribute__((aligned(1024)));

/** RX descriptor ring (aligned to 1KB boundary) */
static grcan_desc_t s_rx_ring[CAN_RX_RING_SIZE]
    __attribute__((aligned(1024)));

/** TX ring write index (software pointer) */
static volatile uint32_t s_tx_wr_idx = 0U;

/** RX ring read index (software pointer) */
static volatile uint32_t s_rx_rd_idx = 0U;

/** CAN node ID */
static uint8_t s_node_id = 0U;

/** CAN base address (GRCAN0 by default) */
static uint32_t s_can_base = GRCAN0_BASE;

/** Bus-off flag */
static volatile uint8_t s_bus_off = 0U;

/** Driver initialization flag */
static volatile uint8_t s_can_initialized = 0U;

/** RX frame software buffer */
static can_frame_t s_rx_sw_buf[CAN_RX_RING_SIZE];
static volatile uint32_t s_rx_sw_wr = 0U;
static volatile uint32_t s_rx_sw_rd = 0U;

/* ======================================================================
 * Forward Declarations
 * ====================================================================== */

static int32_t  can_configure_timing(uint32_t baudrate);
static void     can_process_rx(void);
static void     can_handle_busoff(void);
static void     can_desc_to_frame(const grcan_desc_t *desc, can_frame_t *frame);
static void     can_frame_to_desc(const can_frame_t *frame, grcan_desc_t *desc);

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Initialize the GRCAN controller.
 * @param[in] node_id  CAN node ID (1-127).
 * @param[in] baudrate CAN baudrate in bps.
 * @return CAN_OK on success, negative error code on failure.
 */
int32_t can_init(uint8_t node_id, uint32_t baudrate)
{
    int32_t rc;

    if ((node_id == 0U) || (node_id > 127U)) {
        return CAN_ERR_PARAM;
    }
    if (baudrate == 0U) {
        return CAN_ERR_PARAM;
    }

    s_node_id = node_id;
    s_can_base = GRCAN0_BASE;

    /* Step 1: Reset the GRCAN core */
    REG_WRITE(s_can_base, GRCAN_CTRL, GRCAN_CTRL_RESET);

    /* Wait for reset to complete */
    uint32_t timeout = 10000U;
    while (((REG_READ(s_can_base, GRCAN_CTRL) & GRCAN_CTRL_RESET) != 0U) &&
           (timeout > 0U)) {
        timeout--;
    }
    if (timeout == 0U) {
        return CAN_ERR_INIT;
    }

    /* Step 2: Configure bit timing for requested baudrate */
    rc = can_configure_timing(baudrate);
    if (rc != CAN_OK) {
        return rc;
    }

    /* Step 3: Clear all status flags */
    REG_WRITE(s_can_base, GRCAN_STAT, 0x00000000U);

    /* Step 4: Initialize TX descriptor ring */
    (void)memset(s_tx_ring, 0, sizeof(s_tx_ring));
    s_tx_wr_idx = 0U;

    REG_WRITE(s_can_base, GRCAN_TX_ADDR, (uint32_t)s_tx_ring);
    REG_WRITE(s_can_base, GRCAN_TX_SIZE, (CAN_TX_RING_SIZE * sizeof(grcan_desc_t)) - 1U);
    REG_WRITE(s_can_base, GRCAN_TX_WR, 0U);
    REG_WRITE(s_can_base, GRCAN_TX_RD, 0U);

    /* Step 5: Initialize RX descriptor ring */
    (void)memset(s_rx_ring, 0, sizeof(s_rx_ring));
    s_rx_rd_idx = 0U;

    REG_WRITE(s_can_base, GRCAN_RX_ADDR, (uint32_t)s_rx_ring);
    REG_WRITE(s_can_base, GRCAN_RX_SIZE, (CAN_RX_RING_SIZE * sizeof(grcan_desc_t)) - 1U);
    REG_WRITE(s_can_base, GRCAN_RX_WR, 0U);
    REG_WRITE(s_can_base, GRCAN_RX_RD, 0U);

    /* Step 6: Initialize software RX buffer */
    s_rx_sw_wr = 0U;
    s_rx_sw_rd = 0U;

    /* Step 7: Enable TX and RX DMA */
    REG_WRITE(s_can_base, GRCAN_TX_CTRL, GRCAN_TXRX_ENABLE);
    REG_WRITE(s_can_base, GRCAN_RX_CTRL, GRCAN_TXRX_ENABLE);

    /* Step 8: Configure and enable interrupts */
    /* Enable: RX complete, TX complete, bus-off, error passive, overrun */
    uint32_t irq_mask = 0x0000003FU; /* All primary IRQ sources */
    REG_WRITE(s_can_base, GRCAN_PIMR, irq_mask);

    /* Register ISR */
    rc = irq_register(IRQ_GRCAN0, can_isr);
    if (rc != 0) {
        return CAN_ERR_INIT;
    }
    (void)irq_enable(IRQ_GRCAN0);

    /* Step 9: Enable the CAN core */
    REG_WRITE(s_can_base, GRCAN_CTRL, GRCAN_CTRL_ENABLE);

    s_bus_off = 0U;
    s_can_initialized = 1U;

    return CAN_OK;
}

/**
 * @brief Send a CAN frame.
 * @param[in] cob_id COB-ID.
 * @param[in] data   Data payload.
 * @param[in] len    Data length (0-8).
 * @return CAN_OK on success, negative error code on failure.
 */
int32_t can_send(uint32_t cob_id, const uint8_t *data, uint8_t len)
{
    uint32_t tx_wr;
    uint32_t tx_rd;
    uint32_t next_wr;
    can_frame_t frame;
    grcan_desc_t desc;

    if (s_can_initialized == 0U) {
        return CAN_ERR_INIT;
    }
    if (len > CAN_MAX_DATA_LEN) {
        return CAN_ERR_PARAM;
    }
    if ((data == NULL) && (len > 0U)) {
        return CAN_ERR_PARAM;
    }
    if (s_bus_off != 0U) {
        return CAN_ERR_BUSOFF;
    }

    /* Check TX ring space */
    tx_wr = REG_READ(s_can_base, GRCAN_TX_WR);
    tx_rd = REG_READ(s_can_base, GRCAN_TX_RD);
    next_wr = (tx_wr + sizeof(grcan_desc_t)) %
              (CAN_TX_RING_SIZE * sizeof(grcan_desc_t));

    if (next_wr == tx_rd) {
        return CAN_ERR_FULL; /* TX ring full */
    }

    /* Build frame */
    (void)memset(&frame, 0, sizeof(frame));
    frame.id = cob_id;
    frame.dlc = len;
    frame.extended = (cob_id > CAN_STD_ID_MASK) ? 1U : 0U;
    frame.rtr = 0U;

    if ((data != NULL) && (len > 0U)) {
        (void)memcpy(frame.data, data, len);
    }

    /* Convert to hardware descriptor */
    can_frame_to_desc(&frame, &desc);

    /* Write descriptor to TX ring */
    uint32_t idx = tx_wr / sizeof(grcan_desc_t);
    s_tx_ring[idx] = desc;

    /* Advance TX write pointer (triggers DMA) */
    REG_WRITE(s_can_base, GRCAN_TX_WR, next_wr);

    return CAN_OK;
}

/**
 * @brief Receive a CAN frame with timeout.
 * @param[out] frame      Frame structure to fill.
 * @param[in]  timeout_ms Maximum wait time.
 * @return CAN_OK on success, CAN_ERR_TIMEOUT on timeout.
 */
int32_t can_receive(can_frame_t *frame, uint32_t timeout_ms)
{
    uint32_t start_ms;

    if (frame == NULL) {
        return CAN_ERR_PARAM;
    }
    if (s_can_initialized == 0U) {
        return CAN_ERR_INIT;
    }

    start_ms = bsp_get_uptime_ms();

    /* Poll software RX buffer for available frame */
    while (1) {
        /* Check software buffer */
        uint32_t rd = s_rx_sw_rd;
        uint32_t wr = s_rx_sw_wr;

        if (rd != wr) {
            /* Frame available */
            *frame = s_rx_sw_buf[rd % CAN_RX_RING_SIZE];
            s_rx_sw_rd = (rd + 1U) % CAN_RX_RING_SIZE;
            return CAN_OK;
        }

        /* Also try to process any pending HW RX descriptors */
        can_process_rx();

        /* Check timeout */
        uint32_t elapsed = bsp_get_uptime_ms() - start_ms;
        if (elapsed >= timeout_ms) {
            return CAN_ERR_TIMEOUT;
        }
    }
}

/**
 * @brief CAN interrupt service routine.
 */
void can_isr(void)
{
    uint32_t pimsr;

    pimsr = REG_READ(s_can_base, GRCAN_PISR);

    /* RX interrupt — process received frames */
    if ((pimsr & 0x01U) != 0U) {
        can_process_rx();
    }

    /* TX interrupt — nothing needed, TX is fire-and-forget */

    /* Bus-off / error passive */
    if ((pimsr & 0x04U) != 0U) {
        can_handle_busoff();
    }

    /* Overrun — log error */
    if ((pimsr & 0x10U) != 0U) {
        /* Overrun condition — increment error counter */
        /* FDIR will handle this via health monitoring */
    }

    /* Clear all handled interrupt flags */
    REG_WRITE(s_can_base, GRCAN_PICR, pimsr);
}

/**
 * @brief Check if bus is in bus-off condition.
 * @return 1 if bus-off, 0 if active.
 */
int32_t can_is_bus_off(void)
{
    return (int32_t)s_bus_off;
}

/**
 * @brief Get CAN error counters.
 * @param[out] tx_errors TX error count.
 * @param[out] rx_errors RX error count.
 * @return CAN_OK on success.
 */
int32_t can_get_error_counters(uint32_t *tx_errors, uint32_t *rx_errors)
{
    uint32_t stat;

    if ((tx_errors == NULL) || (rx_errors == NULL)) {
        return CAN_ERR_PARAM;
    }

    stat = REG_READ(s_can_base, GRCAN_STAT);
    /* Error counters in status register upper bytes (device-specific) */
    *tx_errors = (stat >> 16) & 0xFFU;
    *rx_errors = (stat >> 24) & 0xFFU;

    return CAN_OK;
}

/**
 * @brief Get number of frames in software RX buffer.
 * @return Number of pending RX frames.
 */
uint32_t can_rx_pending(void)
{
    uint32_t wr = s_rx_sw_wr;
    uint32_t rd = s_rx_sw_rd;

    if (wr >= rd) {
        return wr - rd;
    }
    return CAN_RX_RING_SIZE - rd + wr;
}

/**
 * @brief Get available TX slots.
 * @return Number of free TX descriptor slots.
 */
uint32_t can_tx_available(void)
{
    uint32_t tx_wr = REG_READ(s_can_base, GRCAN_TX_WR);
    uint32_t tx_rd = REG_READ(s_can_base, GRCAN_TX_RD);
    uint32_t ring_bytes = CAN_TX_RING_SIZE * (uint32_t)sizeof(grcan_desc_t);

    uint32_t used;
    if (tx_wr >= tx_rd) {
        used = tx_wr - tx_rd;
    } else {
        used = ring_bytes - tx_rd + tx_wr;
    }

    uint32_t used_slots = used / (uint32_t)sizeof(grcan_desc_t);
    return CAN_TX_RING_SIZE - used_slots - 1U;
}

/* ======================================================================
 * Private Functions
 * ====================================================================== */

/**
 * @brief Configure CAN bit timing registers.
 * @param[in] baudrate Target baudrate in bps.
 * @return CAN_OK on success.
 */
static int32_t can_configure_timing(uint32_t baudrate)
{
    /*
     * GRCAN bit timing configuration:
     * The CAN bit time = SYNC + PROP + PHASE1 + PHASE2
     * Typically 8-25 time quanta (TQ) per bit.
     *
     * System clock: 50 MHz
     * Target: 1 Mbps → TQ = 50 ns, 10 TQ/bit
     *   SCALER = (50 MHz / (2 * 1 Mbps * 10)) - 1 = 1
     *   PHASE1 = 3, PHASE2 = 3, PROP = 2, SJW = 1
     *
     * For 500 kbps: SCALER = 4
     * For 250 kbps: SCALER = 9
     * For 125 kbps: SCALER = 19
     */

    uint32_t total_tq = 10U; /* Time quanta per bit */
    uint32_t scaler;
    uint32_t conf_val;

    if (baudrate == 0U) {
        return CAN_ERR_PARAM;
    }

    scaler = (SYS_CLK_HZ / (2U * baudrate * total_tq)) - 1U;

    /* Build GRCAN_CONF register:
     * [7:0]   = SCALER
     * [11:8]  = PHASE1 (3)
     * [15:12] = PHASE2 (3)
     * [19:16] = PROP (2)
     * [23:20] = SJW (1)
     * [24]    = SAM (single sample)
     * [25]    = BPR (bus prescaler)
     * [26]    = Selection (1=CAN 2.0B)
     */
    conf_val = (scaler & 0xFFU);
    conf_val |= (3U << 8);   /* PHASE1 = 3 */
    conf_val |= (3U << 12);  /* PHASE2 = 3 */
    conf_val |= (2U << 16);  /* PROP = 2 */
    conf_val |= (1U << 20);  /* SJW = 1 */
    conf_val |= (1U << 26);  /* CAN 2.0B extended frames */

    REG_WRITE(s_can_base, GRCAN_CONF, conf_val);

    return CAN_OK;
}

/**
 * @brief Process received CAN frames from hardware ring to software buffer.
 */
static void can_process_rx(void)
{
    uint32_t rx_wr_hw;
    uint32_t rx_rd_hw;
    uint32_t desc_size = (uint32_t)sizeof(grcan_desc_t);

    rx_wr_hw = REG_READ(s_can_base, GRCAN_RX_WR);
    rx_rd_hw = REG_READ(s_can_base, GRCAN_RX_RD);

    while (rx_rd_hw != rx_wr_hw) {
        uint32_t idx = rx_rd_hw / desc_size;
        can_frame_t frame;

        can_desc_to_frame(&s_rx_ring[idx], &frame);

        /* Store in software buffer if space available */
        uint32_t next_wr = (s_rx_sw_wr + 1U) % CAN_RX_RING_SIZE;
        if (next_wr != s_rx_sw_rd) {
            s_rx_sw_buf[s_rx_sw_wr] = frame;
            s_rx_sw_wr = next_wr;
        }
        /* else: software buffer full, frame dropped */

        /* Advance hardware read pointer */
        rx_rd_hw = (rx_rd_hw + desc_size) %
                   (CAN_RX_RING_SIZE * desc_size);
    }

    /* Update hardware RX read pointer */
    REG_WRITE(s_can_base, GRCAN_RX_RD, rx_rd_hw);
}

/**
 * @brief Handle CAN bus-off condition with automatic recovery.
 */
static void can_handle_busoff(void)
{
    uint32_t stat = REG_READ(s_can_base, GRCAN_STAT);

    if ((stat & GRCAN_STAT_OFF) != 0U) {
        s_bus_off = 1U;

        /* Automatic bus-off recovery:
         * 1. Disable CAN core
         * 2. Clear status
         * 3. Re-enable CAN core
         * The CAN controller will automatically attempt 128 sequences
         * of 11 recessive bits to recover.
         */
        REG_CLR(s_can_base, GRCAN_CTRL, GRCAN_CTRL_ENABLE);

        /* Clear bus-off and error passive flags */
        REG_WRITE(s_can_base, GRCAN_STAT, 0x00000000U);

        /* Small delay (spin-wait) */
        volatile uint32_t delay;
        for (delay = 0U; delay < 1000U; delay++) {
            __asm__ volatile ("nop");
        }

        /* Re-enable CAN core */
        REG_SET(s_can_base, GRCAN_CTRL, GRCAN_CTRL_ENABLE);

        /* Bus-off flag will be cleared when controller enters active state */
        stat = REG_READ(s_can_base, GRCAN_STAT);
        if ((stat & GRCAN_STAT_ACTIVE) != 0U) {
            s_bus_off = 0U;
        }
    }
}

/**
 * @brief Convert hardware descriptor to CAN frame structure.
 * @param[in]  desc  Hardware descriptor.
 * @param[out] frame CAN frame to fill.
 */
static void can_desc_to_frame(const grcan_desc_t *desc, can_frame_t *frame)
{
    if ((desc == NULL) || (frame == NULL)) {
        return;
    }

    (void)memset(frame, 0, sizeof(can_frame_t));

    /* Extract ID and format */
    if ((desc->word0 & DESC_W0_EXT_BIT) != 0U) {
        frame->extended = 1U;
        frame->id = desc->word0 & DESC_W0_ID_EXT_MASK;
    } else {
        frame->extended = 0U;
        frame->id = (desc->word0 & DESC_W0_ID_STD_MASK) >> DESC_W0_ID_STD_SHIFT;
    }

    /* Extract DLC and RTR */
    frame->dlc = (uint8_t)(desc->word1 & DESC_W1_DLC_MASK);
    frame->rtr = ((desc->word1 & DESC_W1_RTR_BIT) != 0U) ? 1U : 0U;

    /* Extract data bytes (big-endian in descriptor) */
    if (frame->dlc > 0U) {
        frame->data[0] = (uint8_t)(desc->data_hi >> 24);
        frame->data[1] = (uint8_t)(desc->data_hi >> 16);
        frame->data[2] = (uint8_t)(desc->data_hi >> 8);
        frame->data[3] = (uint8_t)(desc->data_hi);
    }
    if (frame->dlc > 4U) {
        frame->data[4] = (uint8_t)(desc->data_lo >> 24);
        frame->data[5] = (uint8_t)(desc->data_lo >> 16);
        frame->data[6] = (uint8_t)(desc->data_lo >> 8);
        frame->data[7] = (uint8_t)(desc->data_lo);
    }

    /* Clamp DLC to max */
    if (frame->dlc > CAN_MAX_DATA_LEN) {
        frame->dlc = CAN_MAX_DATA_LEN;
    }
}

/**
 * @brief Convert CAN frame structure to hardware descriptor.
 * @param[in]  frame CAN frame.
 * @param[out] desc  Hardware descriptor to fill.
 */
static void can_frame_to_desc(const can_frame_t *frame, grcan_desc_t *desc)
{
    if ((frame == NULL) || (desc == NULL)) {
        return;
    }

    (void)memset(desc, 0, sizeof(grcan_desc_t));

    /* Set ID and format */
    if (frame->extended != 0U) {
        desc->word0 = DESC_W0_EXT_BIT | (frame->id & DESC_W0_ID_EXT_MASK);
    } else {
        desc->word0 = (frame->id & 0x7FFU) << DESC_W0_ID_STD_SHIFT;
    }

    /* Set DLC and RTR */
    desc->word1 = (uint32_t)(frame->dlc & DESC_W1_DLC_MASK);
    if (frame->rtr != 0U) {
        desc->word1 |= DESC_W1_RTR_BIT;
    }

    /* Set data bytes (big-endian) */
    desc->data_hi = ((uint32_t)frame->data[0] << 24) |
                    ((uint32_t)frame->data[1] << 16) |
                    ((uint32_t)frame->data[2] << 8) |
                    ((uint32_t)frame->data[3]);
    desc->data_lo = ((uint32_t)frame->data[4] << 24) |
                    ((uint32_t)frame->data[5] << 16) |
                    ((uint32_t)frame->data[6] << 8) |
                    ((uint32_t)frame->data[7]);
}
