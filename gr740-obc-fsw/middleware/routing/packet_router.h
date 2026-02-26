/**
 * @file packet_router.h
 * @brief Packet Router — TC dispatch and TM routing.
 *
 * Central hub for incoming telecommand routing to PUS service handlers
 * and outgoing telemetry routing to downlink interfaces (SpW / CAN / UART).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * ECSS-E-ST-70-41C PUS-C compliant routing.
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PACKET_ROUTER_H
#define PACKET_ROUTER_H

#include <stdint.h>
#include "../ccsds/space_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define ROUTER_OK               0
#define ROUTER_ERR_PARAM       -1
#define ROUTER_ERR_NO_HANDLER  -2
#define ROUTER_ERR_CRC         -3
#define ROUTER_ERR_DOWNLINK    -4
#define ROUTER_ERR_QUEUE_FULL  -5

/* ── Configuration ─────────────────────────────────────────────────────── */
#define ROUTER_MAX_SERVICES     24U   /**< Max PUS service type handlers       */
#define ROUTER_TM_QUEUE_DEPTH   64U   /**< Outbound TM queue depth             */

/* ── Downlink path identifiers ─────────────────────────────────────────── */
typedef enum {
    ROUTER_DL_SPW       = 0,   /**< SpaceWire primary downlink             */
    ROUTER_DL_CAN       = 1,   /**< CAN secondary downlink                 */
    ROUTER_DL_UART      = 2,   /**< UART debug/engineering link             */
    ROUTER_DL_COUNT     = 3
} router_downlink_t;

/* ── TC handler callback ───────────────────────────────────────────────── */
/**
 * @brief Callback type for PUS TC handlers.
 *
 * @param[in] svc_subtype  Service subtype from PUS secondary header.
 * @param[in] data         Pointer to TC data field (after secondary header).
 * @param[in] data_len     Length of TC data field in bytes.
 * @return 0 on success, negative on error.
 */
typedef int32_t (*router_tc_handler_t)(uint8_t svc_subtype,
                                       const uint8_t *data,
                                       uint16_t data_len);

/* ── Downlink output callback ──────────────────────────────────────────── */
/**
 * @brief Callback type for TM downlink output.
 *
 * @param[in] frame   Serialised TM packet bytes.
 * @param[in] len     Length in bytes.
 * @return 0 on success, negative on error.
 */
typedef int32_t (*router_downlink_fn_t)(const uint8_t *frame, uint16_t len);

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the packet router.
 * @return ROUTER_OK on success.
 */
int32_t router_init(void);

/**
 * @brief Register a PUS service handler.
 *
 * @param[in] svc_type   PUS service type (1–255).
 * @param[in] handler    TC handler callback.
 * @return ROUTER_OK on success.
 */
int32_t router_register_service(uint8_t svc_type, router_tc_handler_t handler);

/**
 * @brief Register a downlink output path.
 *
 * @param[in] dl_path   Downlink identifier.
 * @param[in] fn        Output callback.
 * @return ROUTER_OK on success.
 */
int32_t router_register_downlink(router_downlink_t dl_path,
                                  router_downlink_fn_t fn);

/**
 * @brief Set the active downlink path.
 * @param[in] dl_path   Which downlink to use for TM output.
 * @return ROUTER_OK on success.
 */
int32_t router_set_active_downlink(router_downlink_t dl_path);

/**
 * @brief Dispatch a received TC packet to the appropriate PUS handler.
 *
 * Validates CCSDS header, checks CRC, extracts PUS secondary header,
 * and invokes the registered handler for the service type.
 *
 * @param[in] raw   Raw TC bytes.
 * @param[in] len   Length in bytes.
 * @return ROUTER_OK on success.
 */
int32_t router_dispatch_tc(const uint8_t *raw, uint16_t len);

/**
 * @brief Enqueue a TM packet for downlink.
 *
 * Called by PUS services (this is the implementation of the extern
 * `router_send_tm` referenced by all PUS service modules).
 *
 * @param[in] pkt   Pointer to assembled CCSDS packet.
 * @return ROUTER_OK on success; ROUTER_ERR_QUEUE_FULL if queue overflow.
 */
int32_t router_send_tm(const ccsds_packet_t *pkt);

/**
 * @brief Process the TM output queue.
 *
 * Serialises queued TM packets and sends them via the active downlink.
 * Call this periodically (e.g. every minor frame).
 *
 * @return Number of packets transmitted, or negative on error.
 */
int32_t router_process_tm_queue(void);

/**
 * @brief Get router statistics.
 *
 * @param[out] tc_count   Total TCs dispatched.
 * @param[out] tm_count   Total TMs sent.
 * @param[out] err_count  Total errors.
 */
void router_get_stats(uint32_t *tc_count, uint32_t *tm_count,
                      uint32_t *err_count);

#ifdef __cplusplus
}
#endif

#endif /* PACKET_ROUTER_H */
