/**
 * @file pus_st05.h
 * @brief PUS Service 5 — Event Reporting (ECSS-E-ST-70-41C).
 *
 * Generates event TM reports with severity levels (info, low, medium, high).
 * Supports event enable/disable and rate limiting.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST05_H
#define PUS_ST05_H

#include <stdint.h>

/** PUS ST05 subtypes */
#define PUS_ST05_INFO       1U  /**< TM(5,1) Informational event   */
#define PUS_ST05_LOW        2U  /**< TM(5,2) Low severity          */
#define PUS_ST05_MEDIUM     3U  /**< TM(5,3) Medium severity       */
#define PUS_ST05_HIGH       4U  /**< TM(5,4) High severity         */

/** Event severity levels */
#define EVT_SEV_INFO        0U
#define EVT_SEV_LOW         1U
#define EVT_SEV_MEDIUM      2U
#define EVT_SEV_HIGH        3U

/** Max event definitions */
#define PUS_ST05_MAX_EVENTS 64U

/** Max auxiliary data per event */
#define PUS_ST05_MAX_AUX    16U

/** Return codes */
#define PUS_ST05_OK             0
#define PUS_ST05_ERR_PARAM      (-1)
#define PUS_ST05_ERR_FULL       (-2)
#define PUS_ST05_ERR_DISABLED   (-3)
#define PUS_ST05_ERR_RATE       (-4)

/**
 * @brief Initialize PUS Service 5.
 * @param[in] apid TM source APID for event reports.
 * @return PUS_ST05_OK on success.
 */
int32_t pus_st05_init(uint16_t apid);

/**
 * @brief Raise an event.
 * @param[in] event_id Event definition ID.
 * @param[in] severity Event severity (EVT_SEV_*).
 * @param[in] aux_data Auxiliary data (may be NULL).
 * @param[in] aux_len  Auxiliary data length.
 * @return PUS_ST05_OK on success.
 */
int32_t pus_st05_raise(uint16_t event_id, uint8_t severity,
                         const uint8_t *aux_data, uint8_t aux_len);

/**
 * @brief Enable event reporting for an event ID.
 * @param[in] event_id Event ID.
 * @return PUS_ST05_OK on success.
 */
int32_t pus_st05_enable(uint16_t event_id);

/**
 * @brief Disable event reporting for an event ID.
 * @param[in] event_id Event ID.
 * @return PUS_ST05_OK on success.
 */
int32_t pus_st05_disable(uint16_t event_id);

/**
 * @brief Get event counters.
 * @param[out] info_cnt   Number of info events raised.
 * @param[out] low_cnt    Number of low events raised.
 * @param[out] med_cnt    Number of medium events raised.
 * @param[out] high_cnt   Number of high events raised.
 */
void pus_st05_get_counters(uint32_t *info_cnt, uint32_t *low_cnt,
                            uint32_t *med_cnt, uint32_t *high_cnt);

#endif /* PUS_ST05_H */
