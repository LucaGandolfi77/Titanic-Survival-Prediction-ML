/**
 * @file task_table.h
 * @brief Task Table — RTEMS task definitions and management.
 *
 * Defines static task configuration table for all OBC tasks with
 * priorities, stack sizes, and entry points for RTEMS task creation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef TASK_TABLE_H
#define TASK_TABLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define TASK_OK             0
#define TASK_ERR_PARAM     -1
#define TASK_ERR_FULL      -2

/* ── Configuration ─────────────────────────────────────────────────────── */
#define TASK_MAX_TASKS      16U
#define TASK_NAME_LEN       8U    /**< RTEMS 4-char names + NUL           */

/* ── Task priorities (lower value = higher RTEMS priority) ─────────────── */
#define TASK_PRI_ISR_DEFER      5U
#define TASK_PRI_SCHEDULER     10U
#define TASK_PRI_TC_HANDLER    15U
#define TASK_PRI_TM_OUTPUT     20U
#define TASK_PRI_HOUSEKEEPING  25U
#define TASK_PRI_FDIR          30U
#define TASK_PRI_THERMAL       40U
#define TASK_PRI_PAYLOAD       50U
#define TASK_PRI_IDLE         250U

/* ── Stack sizes ───────────────────────────────────────────────────────── */
#define TASK_STACK_DEFAULT   (4U * 1024U)   /**< 4 KB default             */
#define TASK_STACK_LARGE     (8U * 1024U)   /**< 8 KB for heavy tasks     */
#define TASK_STACK_SMALL     (2U * 1024U)   /**< 2 KB for lightweight     */

/* ── Task state ────────────────────────────────────────────────────────── */
typedef enum {
    TASK_STATE_DORMANT   = 0,
    TASK_STATE_READY     = 1,
    TASK_STATE_RUNNING   = 2,
    TASK_STATE_SUSPENDED = 3,
    TASK_STATE_BLOCKED   = 4
} task_state_t;

/* ── Task entry point ──────────────────────────────────────────────────── */
typedef void (*task_entry_fn_t)(uint32_t arg);

/* ── Task descriptor ───────────────────────────────────────────────────── */
typedef struct {
    char            name[TASK_NAME_LEN];
    task_entry_fn_t entry;
    uint32_t        priority;
    uint32_t        stack_size;
    uint32_t        period_ms;    /**< 0 = aperiodic (event-driven)       */
    uint32_t        task_id;      /**< RTEMS task ID after creation        */
    task_state_t    state;
    uint8_t         active;
} task_desc_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the task table.
 * @return TASK_OK on success.
 */
int32_t task_table_init(void);

/**
 * @brief Register a task in the table.
 *
 * @param[in] name       Task name (max 7 chars).
 * @param[in] entry      Task entry point.
 * @param[in] priority   RTEMS priority.
 * @param[in] stack_size Stack size in bytes.
 * @param[in] period_ms  Execution period (0 = aperiodic).
 * @return Task index on success, negative on error.
 */
int32_t task_table_register(const char *name,
                             task_entry_fn_t entry,
                             uint32_t priority,
                             uint32_t stack_size,
                             uint32_t period_ms);

/**
 * @brief Get task descriptor by index.
 *
 * @param[in]  index  Task index.
 * @param[out] desc   Pointer to descriptor output.
 * @return TASK_OK on success.
 */
int32_t task_table_get(uint32_t index, task_desc_t *desc);

/**
 * @brief Get total number of registered tasks.
 * @return Task count.
 */
uint32_t task_table_count(void);

/**
 * @brief Update task state.
 *
 * @param[in] index  Task index.
 * @param[in] state  New state.
 * @return TASK_OK on success.
 */
int32_t task_table_set_state(uint32_t index, task_state_t state);

/**
 * @brief Set RTEMS task ID after task creation.
 *
 * @param[in] index   Task index.
 * @param[in] task_id RTEMS task ID.
 * @return TASK_OK on success.
 */
int32_t task_table_set_id(uint32_t index, uint32_t task_id);

#ifdef __cplusplus
}
#endif

#endif /* TASK_TABLE_H */
