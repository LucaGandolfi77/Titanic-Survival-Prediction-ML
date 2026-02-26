/**
 * @file grspw.h
 * @brief GRSPW2 SpaceWire driver interface for GR740.
 *
 * @standard ECSS-E-ST-50-12C (SpaceWire)
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef GRSPW_H
#define GRSPW_H

#include <stdint.h>

/* ======================================================================
 * Return Codes
 * ====================================================================== */
#define SPW_OK                  0       /**< Success                     */
#define SPW_ERR_PARAM           (-1)    /**< Invalid parameter           */
#define SPW_ERR_INIT            (-2)    /**< Initialization failure      */
#define SPW_ERR_TIMEOUT         (-3)    /**< Timeout                     */
#define SPW_ERR_LINK            (-4)    /**< Link error / disconnected   */
#define SPW_ERR_DMA             (-5)    /**< DMA error                   */
#define SPW_ERR_FULL            (-6)    /**< TX buffer full              */
#define SPW_ERR_CREDIT          (-7)    /**< Credit error                */

/* ======================================================================
 * Constants
 * ====================================================================== */
#define SPW_PORT_COUNT          4U      /**< Number of SpW ports on GR740 */
#define SPW_MAX_PORTS           4U

/* ======================================================================
 * Types
 * ====================================================================== */

/** SpaceWire link state enumeration */
typedef enum {
    SPW_LS_ERROR_RESET = 0,     /**< Error-Reset state                   */
    SPW_LS_ERROR_WAIT  = 1,     /**< Error-Wait state                    */
    SPW_LS_READY       = 2,     /**< Ready state                         */
    SPW_LS_STARTED     = 3,     /**< Started state                       */
    SPW_LS_CONNECTING  = 4,     /**< Connecting state                    */
    SPW_LS_RUN         = 5      /**< Run state (link operational)        */
} spw_link_state_t;

/** SpaceWire link status structure */
typedef struct {
    spw_link_state_t state;     /**< Current link state                  */
    uint32_t tx_packets;        /**< TX packet counter                   */
    uint32_t rx_packets;        /**< RX packet counter                   */
    uint32_t credit_errors;     /**< Credit error counter                */
    uint32_t parity_errors;     /**< Parity error counter                */
    uint32_t disconnect_errors; /**< Disconnect error counter            */
    uint32_t escape_errors;     /**< Escape error counter                */
} spw_status_t;

/* ======================================================================
 * Function Prototypes
 * ====================================================================== */

/**
 * @brief Initialize a SpaceWire port.
 * @param[in] port           Port number (0–3).
 * @param[in] link_speed_mbps Link speed in Mbps (10, 50, 100, 200).
 * @return SPW_OK on success, negative error on failure.
 */
int32_t spw_init(uint8_t port, uint32_t link_speed_mbps);

/**
 * @brief Send a packet over SpaceWire.
 * @param[in] port Port number (0–3).
 * @param[in] data Pointer to packet data.
 * @param[in] len  Packet length in bytes.
 * @return SPW_OK on success, negative on failure.
 */
int32_t spw_send(uint8_t port, const uint8_t *data, uint32_t len);

/**
 * @brief Receive a packet from SpaceWire (blocking with timeout).
 * @param[in]  port       Port number (0–3).
 * @param[out] buf        Buffer for received packet.
 * @param[in,out] len     In: buffer size, Out: received length.
 * @param[in]  timeout_ms Maximum wait time.
 * @return SPW_OK on success, SPW_ERR_TIMEOUT on timeout.
 */
int32_t spw_receive(uint8_t port, uint8_t *buf, uint32_t *len,
                    uint32_t timeout_ms);

/**
 * @brief Get link status for a SpaceWire port.
 * @param[in]  port   Port number (0–3).
 * @param[out] status Pointer to status structure.
 * @return SPW_OK on success.
 */
int32_t spw_get_status(uint8_t port, spw_status_t *status);

/**
 * @brief Check if a SpaceWire link is in the RUN state.
 * @param[in] port Port number (0–3).
 * @return 1 if link is running, 0 if not, negative on error.
 */
int32_t spw_link_is_running(uint8_t port);

/**
 * @brief SpaceWire ISR (for a specific port).
 * @param[in] port Port number (0-3).
 */
void spw_isr(uint8_t port);

#endif /* GRSPW_H */
