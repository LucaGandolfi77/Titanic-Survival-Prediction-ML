/**
 * @file minor_frame.h
 * @brief Minor/Major Frame Scheduler API.
 *
 * Implements Rate-Monotonic Scheduling (RMS) with 100 ms minor frame
 * and 1 s major frame (10 minor frames per major frame).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef MINOR_FRAME_H
#define MINOR_FRAME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define SCHED_OK             0
#define SCHED_ERR_PARAM     -1
#define SCHED_ERR_OVERFLOW  -2

/* ── Configuration ─────────────────────────────────────────────────────── */
#define SCHED_MINOR_FRAME_MS   100U     /**< Minor frame period (ms)   */
#define SCHED_MAJOR_FRAME_CNT  10U      /**< Minor frames per major    */
#define SCHED_MAJOR_FRAME_MS   (SCHED_MINOR_FRAME_MS * SCHED_MAJOR_FRAME_CNT)

#define SCHED_MAX_SLOTS        32U      /**< Max scheduled entries     */

/* ── Slot rate dividers ────────────────────────────────────────────────── */
typedef enum {
    SCHED_RATE_100MS  =  1U,  /**< Every minor frame            */
    SCHED_RATE_200MS  =  2U,  /**< Every 2nd minor frame        */
    SCHED_RATE_500MS  =  5U,  /**< Every 5th minor frame        */
    SCHED_RATE_1S     = 10U,  /**< Every major frame            */
    SCHED_RATE_5S     = 50U,  /**< Every 5 seconds              */
    SCHED_RATE_10S    = 100U  /**< Every 10 seconds             */
} sched_rate_t;

/* ── Slot callback ─────────────────────────────────────────────────────── */
typedef void (*sched_slot_fn_t)(void);

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the scheduler.
 * @return SCHED_OK on success.
 */
int32_t sched_init(void);

/**
 * @brief Register a periodic slot.
 *
 * @param[in] name       Human-readable name (max 15 chars).
 * @param[in] fn         Callback function.
 * @param[in] rate       Execution rate divider.
 * @param[in] offset     Phase offset in minor frames (0 .. rate-1).
 * @param[in] priority   Slot priority (lower = higher priority).
 * @return SCHED_OK on success.
 */
int32_t sched_register_slot(const char *name,
                             sched_slot_fn_t fn,
                             sched_rate_t rate,
                             uint8_t offset,
                             uint8_t priority);

/**
 * @brief Tick the scheduler.
 *
 * Call this once per minor frame (every 100 ms).
 * Executes all slots whose phase matches the current minor frame.
 *
 * @return Number of slots executed, or negative on error.
 */
int32_t sched_tick(void);

/**
 * @brief Get the current minor frame counter (modulo major frame).
 * @return 0 .. SCHED_MAJOR_FRAME_CNT-1
 */
uint32_t sched_get_minor_frame(void);

/**
 * @brief Get the total major frame count since boot.
 * @return Major frame counter.
 */
uint32_t sched_get_major_frame(void);

/**
 * @brief Check if any slot overran its deadline in the last tick.
 * @return 1 if overrun detected, 0 otherwise.
 */
uint8_t sched_overrun_detected(void);

#ifdef __cplusplus
}
#endif

#endif /* MINOR_FRAME_H */
