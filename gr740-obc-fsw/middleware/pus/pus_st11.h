/**
 * @file pus_st11.h
 * @brief PUS Service 11 — Time-Based Scheduling (ECSS-E-ST-70-41C).
 *
 * Supports scheduling TC packets for future execution based on
 * on-board time. Provides insert, delete, enable/disable, and
 * time-ordered dispatch.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST11_H
#define PUS_ST11_H

#include <stdint.h>
#include "../ccsds/space_packet.h"

/** PUS ST11 subtypes */
#define PUS_ST11_INSERT     4U  /**< TC(11,4) Insert activity     */
#define PUS_ST11_DELETE     5U  /**< TC(11,5) Delete activity     */
#define PUS_ST11_ENABLE     1U  /**< TC(11,1) Enable scheduling   */
#define PUS_ST11_DISABLE    2U  /**< TC(11,2) Disable scheduling  */
#define PUS_ST11_RESET      3U  /**< TC(11,3) Reset schedule      */
#define PUS_ST11_REPORT     17U /**< TM(11,17) Status report      */

/** Max scheduled activities */
#define PUS_ST11_MAX_SCHED  32U

/** Return codes */
#define PUS_ST11_OK             0
#define PUS_ST11_ERR_PARAM      (-1)
#define PUS_ST11_ERR_FULL       (-2)
#define PUS_ST11_ERR_NOT_FOUND  (-3)

/**
 * @brief Scheduled activity entry.
 */
typedef struct {
    uint32_t       exec_time;   /**< Execution time (CUC coarse)     */
    ccsds_packet_t tc_packet;   /**< TC to execute                   */
    uint16_t       request_id;  /**< Unique request ID               */
    uint8_t        valid;       /**< Entry valid                     */
} sched_entry_t;

/**
 * @brief Initialize PUS Service 11.
 * @return PUS_ST11_OK.
 */
int32_t pus_st11_init(void);

/**
 * @brief Insert a time-tagged activity.
 * @param[in] exec_time Execution time (CUC coarse seconds).
 * @param[in] tc_pkt    TC packet to execute at that time.
 * @return PUS_ST11_OK on success.
 */
int32_t pus_st11_insert(uint32_t exec_time, const ccsds_packet_t *tc_pkt);

/**
 * @brief Delete a scheduled activity by request ID.
 * @param[in] request_id Request ID.
 * @return PUS_ST11_OK on success.
 */
int32_t pus_st11_delete(uint16_t request_id);

/**
 * @brief Enable schedule execution.
 */
void pus_st11_enable(void);

/**
 * @brief Disable schedule execution.
 */
void pus_st11_disable(void);

/**
 * @brief Reset (clear) all scheduled activities.
 */
void pus_st11_reset(void);

/**
 * @brief Dispatch tick — execute due activities.
 * @param[in] current_time Current CUC coarse time.
 * @param[in] handler      Function to dispatch each due TC.
 */
void pus_st11_tick(uint32_t current_time,
                    int32_t (*handler)(const ccsds_packet_t *));

/**
 * @brief Get number of pending activities.
 * @return Count of valid scheduled entries.
 */
uint32_t pus_st11_count(void);

#endif /* PUS_ST11_H */
