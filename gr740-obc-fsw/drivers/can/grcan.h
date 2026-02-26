/**
 * @file grcan.h
 * @brief GRCAN CAN 2.0B controller driver interface for GR740.
 *
 * @standard ISO 11898, CAN 2.0B
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef GRCAN_H
#define GRCAN_H

#include <stdint.h>
#include "can_protocol.h"

/* ======================================================================
 * Return Codes
 * ====================================================================== */
#define CAN_OK                  0       /**< Success                     */
#define CAN_ERR_PARAM           (-1)    /**< Invalid parameter           */
#define CAN_ERR_INIT            (-2)    /**< Initialization failed       */
#define CAN_ERR_TIMEOUT         (-3)    /**< Operation timeout           */
#define CAN_ERR_BUSOFF          (-4)    /**< Bus-off condition           */
#define CAN_ERR_FULL            (-5)    /**< TX buffer full              */
#define CAN_ERR_EMPTY           (-6)    /**< RX buffer empty             */
#define CAN_ERR_OVERRUN         (-7)    /**< Buffer overrun              */

/* ======================================================================
 * Function Prototypes
 * ====================================================================== */

/**
 * @brief Initialize the GRCAN controller.
 * @param[in] node_id  CAN node ID for this OBC (1-127).
 * @param[in] baudrate CAN bus baudrate in bps (e.g., 1000000).
 * @return CAN_OK on success, negative error code on failure.
 */
int32_t can_init(uint8_t node_id, uint32_t baudrate);

/**
 * @brief Send a CAN frame.
 * @param[in] cob_id COB-ID (11-bit standard or 29-bit extended).
 * @param[in] data   Pointer to data payload (up to 8 bytes).
 * @param[in] len    Data length (0-8).
 * @return CAN_OK on success, negative error code on failure.
 */
int32_t can_send(uint32_t cob_id, const uint8_t *data, uint8_t len);

/**
 * @brief Receive a CAN frame (blocking with timeout).
 * @param[out] frame      Pointer to frame structure to fill.
 * @param[in]  timeout_ms Maximum time to wait in milliseconds.
 * @return CAN_OK on success, CAN_ERR_TIMEOUT if no frame received,
 *         or other negative error code.
 */
int32_t can_receive(can_frame_t *frame, uint32_t timeout_ms);

/**
 * @brief CAN interrupt service routine.
 *
 * Handles TX completion, RX frame available, and error conditions.
 * Should be registered with the IRQ controller for the GRCAN IRQ.
 */
void can_isr(void);

/**
 * @brief Check if the CAN bus is in bus-off condition.
 * @return 1 if bus-off, 0 if active.
 */
int32_t can_is_bus_off(void);

/**
 * @brief Get CAN error counters.
 * @param[out] tx_errors TX error counter value.
 * @param[out] rx_errors RX error counter value.
 * @return CAN_OK on success.
 */
int32_t can_get_error_counters(uint32_t *tx_errors, uint32_t *rx_errors);

/**
 * @brief Get the number of frames in the RX buffer.
 * @return Number of pending RX frames.
 */
uint32_t can_rx_pending(void);

/**
 * @brief Get the available space in the TX buffer.
 * @return Number of available TX slots.
 */
uint32_t can_tx_available(void);

#endif /* GRCAN_H */
