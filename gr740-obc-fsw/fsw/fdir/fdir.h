/**
 * @file fdir.h
 * @brief Fault Detection, Isolation, and Recovery (FDIR) API.
 *
 * Three-level fault taxonomy:
 *   Level 1: Local recovery (automatic retry/reset)
 *   Level 2: Subsystem isolation (safe mode transition)
 *   Level 3: System recovery (full reset / safe mode)
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef FDIR_H
#define FDIR_H

#include <stdint.h>
#include "error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define FDIR_OK             0
#define FDIR_ERR_PARAM     -1
#define FDIR_ERR_FULL      -2

/* ── Configuration ─────────────────────────────────────────────────────── */
#define FDIR_MAX_FAULTS     64U   /**< Max fault entries     */
#define FDIR_MAX_HISTORY    32U   /**< Fault history depth   */
#define FDIR_RETRY_LIMIT     3U   /**< Auto-retry before escalation */

/* ── FDIR levels ───────────────────────────────────────────────────────── */
typedef enum {
    FDIR_LEVEL_1 = 1,   /**< Local recovery     */
    FDIR_LEVEL_2 = 2,   /**< Subsystem isolate  */
    FDIR_LEVEL_3 = 3    /**< System recovery    */
} fdir_level_t;

/* ── Recovery action callback ──────────────────────────────────────────── */
typedef int32_t (*fdir_recovery_fn_t)(error_code_t err_code, uint32_t aux);

/* ── Fault entry ───────────────────────────────────────────────────────── */
typedef struct {
    error_code_t       err_code;
    fdir_level_t       level;
    fdir_recovery_fn_t recovery_fn;
    uint32_t           occurrence_count;
    uint32_t           last_time_ms;
    uint8_t            enabled;
    uint8_t            active;    /**< Fault currently active */
} fdir_fault_t;

/* ── Fault history record ──────────────────────────────────────────────── */
typedef struct {
    error_code_t  err_code;
    fdir_level_t  level;
    uint32_t      timestamp_ms;
    int32_t       recovery_result;
} fdir_history_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the FDIR module.
 * @return FDIR_OK on success.
 */
int32_t fdir_init(void);

/**
 * @brief Register a fault definition.
 *
 * @param[in] err_code   Error code.
 * @param[in] level      FDIR level.
 * @param[in] recovery   Recovery action callback (may be NULL for L3).
 * @return FDIR_OK on success.
 */
int32_t fdir_register_fault(error_code_t err_code,
                             fdir_level_t level,
                             fdir_recovery_fn_t recovery);

/**
 * @brief Report a fault occurrence.
 *
 * Looks up the fault, increments count, invokes recovery action.
 * If recovery fails FDIR_RETRY_LIMIT times, escalates to next level.
 *
 * @param[in] err_code  Error code.
 * @param[in] aux       Auxiliary data (e.g. register value).
 * @return Recovery result, or negative on lookup failure.
 */
int32_t fdir_report_fault(error_code_t err_code, uint32_t aux);

/**
 * @brief Clear a fault (mark resolved).
 *
 * @param[in] err_code  Error code.
 * @return FDIR_OK on success.
 */
int32_t fdir_clear_fault(error_code_t err_code);

/**
 * @brief Check if a specific fault is currently active.
 *
 * @param[in] err_code  Error code.
 * @return 1 if active, 0 if not, negative on error.
 */
int32_t fdir_is_active(error_code_t err_code);

/**
 * @brief Get the number of active faults.
 * @return Active fault count.
 */
uint32_t fdir_active_count(void);

/* ── Compatibility history view ───────────────────────────────────────── */
typedef struct {
    uint16_t        fault_id;        /**< Error code (16-bit view) */
    uint32_t        timestamp_ms;    /**< Time of occurrence */
    int32_t         recovery_result; /**< Recovery action result */
} fdir_history_entry_t;

/**
 * @brief Get the number of entries in the fault history.
 * @return Number of entries recorded.
 */
uint32_t fdir_get_history_count(void);

/**
 * @brief Get a single history entry by index (0 = most recent).
 *
 * Compatibility function used by the unit test harness. Returns 0 on
 * success or a negative error code.
 */
int32_t fdir_get_history(uint32_t index, fdir_history_entry_t *entry);

/**
 * @brief Periodic FDIR tick — check for persistent faults.
 *
 * Call every minor frame.
 *
 * @return Number of escalations triggered.
 */
int32_t fdir_tick(void);

#ifdef __cplusplus
}
#endif

#endif /* FDIR_H */
