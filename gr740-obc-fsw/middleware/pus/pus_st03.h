/**
 * @file pus_st03.h
 * @brief PUS Service 3 — Housekeeping & Diagnostic Data (ECSS-E-ST-70-41C).
 *
 * Supports configurable HK parameter definitions, periodic HK report
 * generation, and on-demand HK requests.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST03_H
#define PUS_ST03_H

#include <stdint.h>

/** PUS ST03 subtypes */
#define PUS_ST03_HK_REPORT          25U /**< TM(3,25) HK parameter report    */
#define PUS_ST03_DIAG_REPORT        26U /**< TM(3,26) Diagnostic report      */
#define PUS_ST03_ENABLE_PERIODIC    5U  /**< TC(3,5) Enable periodic gen     */
#define PUS_ST03_DISABLE_PERIODIC   6U  /**< TC(3,6) Disable periodic gen    */
#define PUS_ST03_DEFINE_HK         1U   /**< TC(3,1) Create HK report def   */
#define PUS_ST03_DELETE_HK         3U   /**< TC(3,3) Delete HK report def   */
#define PUS_ST03_REQUEST_HK        27U  /**< TC(3,27) Request one-shot HK   */

/** Maximum HK structure definitions */
#define PUS_ST03_MAX_SID           16U  /**< Max structure IDs              */
#define PUS_ST03_MAX_PARAMS        32U  /**< Max parameters per SID         */

/** Return codes */
#define PUS_ST03_OK                0
#define PUS_ST03_ERR_PARAM         (-1)
#define PUS_ST03_ERR_FULL          (-2)
#define PUS_ST03_ERR_NOT_FOUND     (-3)

/** HK parameter source callback: reads value at given parameter ID */
typedef int32_t (*hk_param_reader_t)(uint16_t param_id, uint8_t *buf,
                                      uint32_t *len);

/**
 * @brief HK structure definition.
 */
typedef struct {
    uint16_t sid;                               /**< Structure ID          */
    uint8_t  enabled;                           /**< Periodic gen enabled  */
    uint16_t period_ms;                         /**< Collection period     */
    uint32_t last_time_ms;                      /**< Last generation time  */
    uint16_t param_ids[PUS_ST03_MAX_PARAMS];    /**< Parameter IDs         */
    uint8_t  param_sizes[PUS_ST03_MAX_PARAMS];  /**< Param size in bytes   */
    uint8_t  num_params;                        /**< Number of parameters  */
    uint8_t  valid;                             /**< Definition valid      */
} hk_definition_t;

/**
 * @brief Initialize PUS Service 3.
 * @param[in] apid   TM source APID for HK reports.
 * @param[in] reader Callback to read parameter values.
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_init(uint16_t apid, hk_param_reader_t reader);

/**
 * @brief Define a HK report structure.
 * @param[in] sid        Structure ID.
 * @param[in] param_ids  Array of parameter IDs.
 * @param[in] param_sizes Array of param sizes.
 * @param[in] num_params Number of parameters.
 * @param[in] period_ms  Collection period in ms (0 = on-demand only).
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_define(uint16_t sid, const uint16_t *param_ids,
                         const uint8_t *param_sizes, uint8_t num_params,
                         uint16_t period_ms);

/**
 * @brief Delete a HK report definition.
 * @param[in] sid Structure ID.
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_delete(uint16_t sid);

/**
 * @brief Enable periodic generation for a SID.
 * @param[in] sid Structure ID.
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_enable(uint16_t sid);

/**
 * @brief Disable periodic generation for a SID.
 * @param[in] sid Structure ID.
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_disable(uint16_t sid);

/**
 * @brief Generate and send a one-shot HK report.
 * @param[in] sid Structure ID.
 * @return PUS_ST03_OK on success.
 */
int32_t pus_st03_report(uint16_t sid);

/**
 * @brief Periodic tick — call from scheduler to generate due HK reports.
 * @param[in] current_time_ms Current time in ms.
 */
void pus_st03_tick(uint32_t current_time_ms);

#endif /* PUS_ST03_H */
