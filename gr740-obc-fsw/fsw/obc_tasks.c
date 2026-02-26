/**
 * @file obc_tasks.c
 * @brief OBC Task Creation and entry-point implementations.
 *
 * Each task runs in a periodic or event-driven loop, servicing its
 * watchdog heartbeat at the top of every iteration.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "obc_tasks.h"
#include "scheduler/minor_frame.h"
#include "scheduler/task_table.h"
#include "watchdog/watchdog.h"
#include "watchdog/health_monitor.h"
#include "fdir/fdir.h"
#include "modes/mode_manager.h"
#include "../middleware/routing/packet_router.h"
#include "../middleware/pus/pus_st03.h"
#include "../middleware/pus/pus_st09.h"
#include "../middleware/pus/pus_st11.h"

extern uint32_t bsp_get_uptime_ms(void);
extern void     bsp_delay_ms(uint32_t ms);

/* ── Task entry points ─────────────────────────────────────────────────── */

/**
 * @brief Scheduler task — drives the 100 ms minor frame loop.
 */
void task_scheduler_entry(uint32_t arg)
{
    uint32_t next_tick_ms;
    (void)arg;

    next_tick_ms = bsp_get_uptime_ms() + SCHED_MINOR_FRAME_MS;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_SCHED);

        /* Execute scheduled slots for this minor frame */
        (void)sched_tick();

        /* Kick HW watchdog */
        wdg_hw_kick();

        /* Wait until next minor frame boundary */
        while (bsp_get_uptime_ms() < next_tick_ms) {
            /* Busy-wait or yield (RTEMS: rtems_task_wake_after(1)) */
        }
        next_tick_ms += SCHED_MINOR_FRAME_MS;
    }
}

/**
 * @brief TC receiver task — polls interfaces for incoming TCs.
 */
void task_tc_rx_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_TC_RX);

        /* In RTEMS: would block on semaphore from SpW/CAN/UART ISR.
         * For bare-metal, poll receive buffers.                         */
        /* TODO: Interface-specific receive calls + router_dispatch_tc() */

        bsp_delay_ms(10U);  /* 10 ms polling interval */
    }
}

/**
 * @brief TM transmitter task — flushes TM queue via active downlink.
 */
void task_tm_tx_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_TM_TX);

        (void)router_process_tm_queue();

        bsp_delay_ms(50U);  /* 50 ms drain interval */
    }
}

/**
 * @brief Housekeeping task — collects HK, manages PUS ST03 and time.
 */
void task_housekeeping_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_HK);

        /* Tick PUS services */
        pus_st03_tick();
        pus_st11_tick();

        /* Update health monitor */
        (void)hmon_tick();

        bsp_delay_ms(100U);
    }
}

/**
 * @brief FDIR task — fault monitoring and recovery.
 */
void task_fdir_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_FDIR);

        /* Check watchdog expirations */
        {
            uint32_t expired_id = 0U;
            int32_t  n = wdg_check_all(&expired_id);
            if (n > 0) {
                (void)fdir_report_fault(ERR_OBC_WATCHDOG_TIMEOUT, expired_id);
            }
        }

        /* Check scheduler overruns */
        if (sched_overrun_detected() != 0U) {
            (void)fdir_report_fault(ERR_OBC_TASK_OVERRUN, 0U);
        }

        /* FDIR periodic tick */
        (void)fdir_tick();

        bsp_delay_ms(100U);
    }
}

/**
 * @brief Thermal control task.
 */
void task_thermal_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_THERMAL);

        /* TODO: thermal_control_tick() when thermal module is active */

        bsp_delay_ms(1000U);  /* 1 Hz thermal control loop */
    }
}

/**
 * @brief Payload management task.
 */
void task_payload_entry(uint32_t arg)
{
    (void)arg;

    for (;;) {
        wdg_heartbeat(OBC_TASK_ID_PAYLOAD);

        /* Only active in SCIENCE mode */
        if (mode_get_current() == MODE_SCIENCE) {
            /* TODO: payload data acquisition */
        }

        bsp_delay_ms(500U);
    }
}

/* ── Registration and startup ──────────────────────────────────────────── */

int32_t obc_tasks_register_all(void)
{
    int32_t ret;

    ret = task_table_register("SCHED",   task_scheduler_entry,
                               TASK_PRI_SCHEDULER, TASK_STACK_DEFAULT,
                               SCHED_MINOR_FRAME_MS);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("TC_RX",   task_tc_rx_entry,
                               TASK_PRI_TC_HANDLER, TASK_STACK_DEFAULT, 10U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("TM_TX",   task_tm_tx_entry,
                               TASK_PRI_TM_OUTPUT, TASK_STACK_DEFAULT, 50U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("HK",      task_housekeeping_entry,
                               TASK_PRI_HOUSEKEEPING, TASK_STACK_DEFAULT,
                               100U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("FDIR",    task_fdir_entry,
                               TASK_PRI_FDIR, TASK_STACK_DEFAULT, 100U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("THERM",   task_thermal_entry,
                               TASK_PRI_THERMAL, TASK_STACK_SMALL, 1000U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    ret = task_table_register("PLOAD",   task_payload_entry,
                               TASK_PRI_PAYLOAD, TASK_STACK_DEFAULT, 500U);
    if (ret < 0) { return OBC_TASK_ERR_CREATE; }

    /* Register SW watchdog heartbeats for all tasks */
    (void)wdg_register_task(OBC_TASK_ID_SCHED,   WDG_SW_TIMEOUT_MS, "SCHED");
    (void)wdg_register_task(OBC_TASK_ID_TC_RX,   WDG_SW_TIMEOUT_MS, "TC_RX");
    (void)wdg_register_task(OBC_TASK_ID_TM_TX,   WDG_SW_TIMEOUT_MS, "TM_TX");
    (void)wdg_register_task(OBC_TASK_ID_HK,      WDG_SW_TIMEOUT_MS, "HK");
    (void)wdg_register_task(OBC_TASK_ID_FDIR,    WDG_SW_TIMEOUT_MS, "FDIR");
    (void)wdg_register_task(OBC_TASK_ID_THERMAL, 2000U,              "THERM");
    (void)wdg_register_task(OBC_TASK_ID_PAYLOAD, 2000U,              "PLOAD");

    return OBC_TASK_OK;
}

int32_t obc_tasks_start_all(void)
{
    /* In a real RTEMS build, this would call rtems_task_create()
     * and rtems_task_start() for each registered task.
     *
     * For bare-metal single-thread: the scheduler task is called
     * directly from obc_main and other tasks are dispatched from
     * the minor-frame slot table.
     */
    uint32_t count = task_table_count();
    uint32_t i;

    for (i = 0U; i < count; i++) {
        (void)task_table_set_state(i, TASK_STATE_READY);
    }

    return OBC_TASK_OK;
}
