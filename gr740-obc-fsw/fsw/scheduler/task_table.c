/**
 * @file task_table.c
 * @brief Task Table — static task registry implementation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "task_table.h"

/* ── Module state ──────────────────────────────────────────────────────── */
static task_desc_t  tasks[TASK_MAX_TASKS];
static uint32_t     task_cnt = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t task_table_init(void)
{
    uint32_t i;
    for (i = 0U; i < TASK_MAX_TASKS; i++) {
        tasks[i].entry      = (task_entry_fn_t)0;
        tasks[i].priority   = 0U;
        tasks[i].stack_size = 0U;
        tasks[i].period_ms  = 0U;
        tasks[i].task_id    = 0U;
        tasks[i].state      = TASK_STATE_DORMANT;
        tasks[i].active     = 0U;
        tasks[i].name[0]    = '\0';
    }
    task_cnt = 0U;
    return TASK_OK;
}

int32_t task_table_register(const char *name,
                             task_entry_fn_t entry,
                             uint32_t priority,
                             uint32_t stack_size,
                             uint32_t period_ms)
{
    uint32_t i;
    task_desc_t *t;

    if ((entry == (task_entry_fn_t)0) || (name == (const char *)0)) {
        return TASK_ERR_PARAM;
    }
    if (task_cnt >= TASK_MAX_TASKS) {
        return TASK_ERR_FULL;
    }

    t = &tasks[task_cnt];
    t->entry      = entry;
    t->priority   = priority;
    t->stack_size = stack_size;
    t->period_ms  = period_ms;
    t->task_id    = 0U;
    t->state      = TASK_STATE_DORMANT;
    t->active     = 1U;

    for (i = 0U; (i < (TASK_NAME_LEN - 1U)) && (name[i] != '\0'); i++) {
        t->name[i] = name[i];
    }
    t->name[i] = '\0';

    task_cnt++;
    return (int32_t)(task_cnt - 1U);
}

int32_t task_table_get(uint32_t index, task_desc_t *desc)
{
    if ((index >= task_cnt) || (desc == (task_desc_t *)0)) {
        return TASK_ERR_PARAM;
    }

    *desc = tasks[index];
    return TASK_OK;
}

uint32_t task_table_count(void)
{
    return task_cnt;
}

int32_t task_table_set_state(uint32_t index, task_state_t state)
{
    if (index >= task_cnt) {
        return TASK_ERR_PARAM;
    }
    tasks[index].state = state;
    return TASK_OK;
}

int32_t task_table_set_id(uint32_t index, uint32_t task_id)
{
    if (index >= task_cnt) {
        return TASK_ERR_PARAM;
    }
    tasks[index].task_id = task_id;
    return TASK_OK;
}
