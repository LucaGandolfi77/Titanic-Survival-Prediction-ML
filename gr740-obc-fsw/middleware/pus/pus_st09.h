/**
 * @file pus_st09.h
 * @brief PUS Service 9 — Time Management (ECSS-E-ST-70-41C).
 *
 * Manages on-board time in CUC format (4-byte coarse + 2-byte fine).
 * Supports time setting from ground (TC(9,1)) and time reporting
 * via TM(9,2). Time correlation maintained against hardware timer.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST09_H
#define PUS_ST09_H

#include <stdint.h>

/** PUS ST09 subtypes */
#define PUS_ST09_SET_TIME       1U  /**< TC(9,1) Set on-board time    */
#define PUS_ST09_REPORT_TIME    2U  /**< TM(9,2) Report on-board time */

/** CUC time structure: 4 bytes coarse + 2 bytes fine */
typedef struct {
    uint32_t coarse;    /**< Seconds since epoch        */
    uint16_t fine;      /**< Fractional second (1/65536)*/
} cuc_time_t;

/** Return codes */
#define PUS_ST09_OK         0
#define PUS_ST09_ERR_PARAM  (-1)

/**
 * @brief Initialize PUS Service 9.
 * @param[in] apid TM source APID for time reports.
 * @return PUS_ST09_OK on success.
 */
int32_t pus_st09_init(uint16_t apid);

/**
 * @brief Set on-board time (TC(9,1)).
 * @param[in] time CUC time to set.
 * @return PUS_ST09_OK on success.
 */
int32_t pus_st09_set_time(const cuc_time_t *time);

/**
 * @brief Get current on-board time.
 * @param[out] time Current CUC time.
 * @return PUS_ST09_OK on success.
 */
int32_t pus_st09_get_time(cuc_time_t *time);

/**
 * @brief Generate and send time report TM(9,2).
 * @return PUS_ST09_OK on success.
 */
int32_t pus_st09_report(void);

/**
 * @brief Tick — call every millisecond to advance fine counter.
 */
void pus_st09_tick_ms(void);

/**
 * @brief Process a TC(9,1) data field.
 * @param[in] data Data (6 bytes: coarse(4) + fine(2)).
 * @param[in] len  Data length.
 * @return PUS_ST09_OK on success.
 */
int32_t pus_st09_process_tc(const uint8_t *data, uint32_t len);

#endif /* PUS_ST09_H */
