/**
 * @file obc_tasks.h
 * @brief OBC Task Creation and Management API.
 *
 * Provides RTEMS task creation wrappers and task entry points
 * for all OBC software tasks.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef OBC_TASKS_H
#define OBC_TASKS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define OBC_TASK_OK          0
#define OBC_TASK_ERR_CREATE -1

/* ── Task IDs ──────────────────────────────────────────────────────────── */
#define OBC_TASK_ID_SCHED       1U
#define OBC_TASK_ID_TC_RX       2U
#define OBC_TASK_ID_TM_TX       3U
#define OBC_TASK_ID_HK          4U
#define OBC_TASK_ID_FDIR        5U
#define OBC_TASK_ID_THERMAL     6U
#define OBC_TASK_ID_PAYLOAD     7U

/* ── Task entry points ─────────────────────────────────────────────────── */
void task_scheduler_entry(uint32_t arg);
void task_tc_rx_entry(uint32_t arg);
void task_tm_tx_entry(uint32_t arg);
void task_housekeeping_entry(uint32_t arg);
void task_fdir_entry(uint32_t arg);
void task_thermal_entry(uint32_t arg);
void task_payload_entry(uint32_t arg);

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Register all OBC tasks in the task table.
 * @return OBC_TASK_OK on success.
 */
int32_t obc_tasks_register_all(void);

/**
 * @brief Create and start all registered RTEMS tasks.
 * @return OBC_TASK_OK on success.
 */
int32_t obc_tasks_start_all(void);

#ifdef __cplusplus
}
#endif

#endif /* OBC_TASKS_H */
