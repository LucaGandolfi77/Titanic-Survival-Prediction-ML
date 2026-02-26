/**
 * @file watchdog.h
 * @brief Watchdog Manager — HW and SW watchdog API.
 *
 * Dual-watchdog architecture:
 *   - Hardware WDT: 2 s timeout, kicked from main loop
 *   - Software WDT: 500 ms per-task heartbeat monitoring
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef WATCHDOG_H
#define WATCHDOG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define WDG_OK             0
#define WDG_ERR_PARAM     -1
#define WDG_ERR_TIMEOUT   -2

/* ── Configuration ─────────────────────────────────────────────────────── */
#define WDG_HW_TIMEOUT_MS    2000U   /**< HW watchdog period            */
#define WDG_SW_TIMEOUT_MS     500U   /**< Per-task heartbeat deadline    */
#define WDG_MAX_MONITORED      16U   /**< Max tasks to monitor          */

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the watchdog manager.
 *
 * Configures the hardware watchdog timer (GPTIMER) and prepares
 * the software heartbeat table.
 *
 * @return WDG_OK on success.
 */
int32_t wdg_init(void);

/**
 * @brief Kick (service) the hardware watchdog.
 *
 * Must be called from the main loop at least every WDG_HW_TIMEOUT_MS.
 */
void wdg_hw_kick(void);

/**
 * @brief Enable the hardware watchdog.
 *
 * After calling this, the hardware watchdog must be kicked periodically
 * or the system will reset.
 */
void wdg_hw_enable(void);

/**
 * @brief Disable the hardware watchdog (use only during debug).
 */
void wdg_hw_disable(void);

/**
 * @brief Register a task for software heartbeat monitoring.
 *
 * @param[in] task_id     Unique task identifier.
 * @param[in] timeout_ms  Heartbeat timeout in ms.
 * @param[in] name        Task name (max 15 chars).
 * @return WDG_OK on success.
 */
int32_t wdg_register_task(uint32_t task_id, uint32_t timeout_ms,
                           const char *name);

/**
 * @brief Report heartbeat from a task.
 *
 * @param[in] task_id  Task identifier.
 * @return WDG_OK on success.
 */
int32_t wdg_heartbeat(uint32_t task_id);

/**
 * @brief Check all monitored tasks for heartbeat timeout.
 *
 * Call this periodically (e.g. every minor frame).
 *
 * @param[out] expired_id  Pointer to receive the first expired task ID
 *                         (NULL if not needed).
 * @return Number of timed-out tasks, 0 if all healthy.
 */
int32_t wdg_check_all(uint32_t *expired_id);

#ifdef __cplusplus
}
#endif

#endif /* WATCHDOG_H */
