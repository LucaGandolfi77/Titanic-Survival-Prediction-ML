/**
 * @file health_monitor.c
 * @brief Health Monitor implementation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "health_monitor.h"
#include "watchdog.h"

/* ── Module state ──────────────────────────────────────────────────────── */
static health_status_t  subsys_status[HMON_SUBSYS_COUNT];
static uint8_t          hmon_init_done = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t hmon_init(void)
{
    uint32_t i;

    for (i = 0U; i < (uint32_t)HMON_SUBSYS_COUNT; i++) {
        subsys_status[i] = HEALTH_UNKNOWN;
    }
    hmon_init_done = 1U;
    return HMON_OK;
}

int32_t hmon_set_status(hmon_subsys_t subsys, health_status_t status)
{
    if ((uint32_t)subsys >= (uint32_t)HMON_SUBSYS_COUNT) {
        return HMON_ERR_PARAM;
    }
    subsys_status[subsys] = status;
    return HMON_OK;
}

int32_t hmon_get_status(hmon_subsys_t subsys, health_status_t *status)
{
    if ((uint32_t)subsys >= (uint32_t)HMON_SUBSYS_COUNT) {
        return HMON_ERR_PARAM;
    }
    if (status == (health_status_t *)0) {
        return HMON_ERR_PARAM;
    }
    *status = subsys_status[subsys];
    return HMON_OK;
}

health_status_t hmon_get_overall(void)
{
    uint32_t i;
    health_status_t worst = HEALTH_NOMINAL;

    for (i = 0U; i < (uint32_t)HMON_SUBSYS_COUNT; i++) {
        if ((uint32_t)subsys_status[i] > (uint32_t)worst) {
            worst = subsys_status[i];
        }
    }
    return worst;
}

int32_t hmon_tick(void)
{
    int32_t  expired_count;
    uint32_t expired_id = 0U;
    int32_t  faulty = 0;

    if (hmon_init_done == 0U) {
        return 0;
    }

    /* Check SW watchdog heartbeats */
    expired_count = wdg_check_all(&expired_id);
    if (expired_count > 0) {
        /* Mark OBC subsystem as degraded if any task heartbeat expired */
        subsys_status[HMON_SUBSYS_OBC] = HEALTH_DEGRADED;
    } else {
        if (subsys_status[HMON_SUBSYS_OBC] == HEALTH_DEGRADED) {
            subsys_status[HMON_SUBSYS_OBC] = HEALTH_NOMINAL;
        }
    }

    /* Count faulty subsystems */
    {
        uint32_t i;
        for (i = 0U; i < (uint32_t)HMON_SUBSYS_COUNT; i++) {
            if (subsys_status[i] == HEALTH_FAULTY) {
                faulty++;
            }
        }
    }

    return faulty;
}

uint32_t hmon_get_vector(void)
{
    uint32_t vec = 0U;
    uint32_t i;

    for (i = 0U; i < (uint32_t)HMON_SUBSYS_COUNT; i++) {
        vec |= (((uint32_t)subsys_status[i] & 0x03U) << (i * 2U));
    }
    return vec;
}
