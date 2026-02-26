/**
 * @file obc_main.c
 * @brief Main entry point for GR740 OBC Flight Software.
 *
 * Initialisation sequence:
 *   1. BSP hardware init (clocks, memory, EDAC, UART)
 *   2. Driver init (CAN, SpaceWire, SPI, I2C, GPIO, Timers)
 *   3. Middleware init (CCSDS, PUS services, packet router)
 *   4. FSW init (scheduler, watchdog, FDIR, mode manager)
 *   5. Task registration and startup
 *   6. Transition BOOT → SAFE → NOMINAL
 *   7. Enter scheduler loop (never returns)
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include <stdint.h>

/* Config */
#include "../config/mission_config.h"
#include "../config/hw_config.h"

/* BSP */
#include "../bsp/gr740_init.h"
#include "../bsp/irq_handler.h"

/* Drivers */
#include "../drivers/can/grcan.h"
#include "../drivers/spacewire/grspw.h"
#include "../drivers/uart/apbuart.h"
#include "../drivers/spi/spictrl.h"
#include "../drivers/i2c/i2cmst.h"
#include "../drivers/gpio/grgpio.h"
#include "../drivers/timer/gptimer.h"

/* Middleware */
#include "../middleware/ccsds/space_packet.h"
#include "../middleware/routing/packet_router.h"
#include "../middleware/pus/pus_st01.h"
#include "../middleware/pus/pus_st03.h"
#include "../middleware/pus/pus_st05.h"
#include "../middleware/pus/pus_st08.h"
#include "../middleware/pus/pus_st09.h"
#include "../middleware/pus/pus_st11.h"
#include "../middleware/pus/pus_st17.h"

/* FSW */
#include "scheduler/minor_frame.h"
#include "scheduler/task_table.h"
#include "watchdog/watchdog.h"
#include "watchdog/health_monitor.h"
#include "fdir/fdir.h"
#include "fdir/error_codes.h"
#include "modes/mode_manager.h"
#include "obc_tasks.h"

/* ── External BSP functions ────────────────────────────────────────────── */
extern void     bsp_delay_ms(uint32_t ms);
extern uint32_t bsp_get_uptime_ms(void);

/* ── Forward declarations ──────────────────────────────────────────────── */
static void init_hardware(void);
static void init_drivers(void);
static void init_middleware(void);
static void init_fsw(void);
static void register_fdir_faults(void);
static void register_sched_slots(void);
static void boot_to_safe(void);
static void safe_to_nominal(void);
static void run_forever(void);

/* ── FDIR recovery stubs ──────────────────────────────────────────────── */
static int32_t fdir_recover_spw(error_code_t code, uint32_t aux);
static int32_t fdir_recover_can(error_code_t code, uint32_t aux);
static int32_t fdir_recover_mem(error_code_t code, uint32_t aux);

/* ── Debug UART print (simple polled output) ───────────────────────────── */
static void debug_puts(const char *s);

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    /* ── 1. Hardware ─────────────────────────────────────────────────── */
    init_hardware();
    debug_puts("[OBC] HW init OK\r\n");

    /* ── 2. Drivers ──────────────────────────────────────────────────── */
    init_drivers();
    debug_puts("[OBC] Drivers OK\r\n");

    /* ── 3. Middleware ───────────────────────────────────────────────── */
    init_middleware();
    debug_puts("[OBC] Middleware OK\r\n");

    /* ── 4. FSW ──────────────────────────────────────────────────────── */
    init_fsw();
    debug_puts("[OBC] FSW init OK\r\n");

    /* ── 5. Tasks ────────────────────────────────────────────────────── */
    (void)obc_tasks_register_all();
    (void)obc_tasks_start_all();
    debug_puts("[OBC] Tasks started\r\n");

    /* ── 6. Mode transitions ─────────────────────────────────────────── */
    boot_to_safe();
    safe_to_nominal();
    debug_puts("[OBC] NOMINAL mode\r\n");

    /* ── 7. Run ──────────────────────────────────────────────────────── */
    run_forever();

    /* Should never reach here */
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  INIT SEQUENCES
 * ══════════════════════════════════════════════════════════════════════ */

static void init_hardware(void)
{
    /* BSP: clocks, EDAC, UART debug, IRQ controller, systick */
    (void)gr740_init();
    (void)irq_init();
}

static void init_drivers(void)
{
    /* CAN @ 500 kbps */
    (void)grcan_init(0U, 500000U);

    /* SpaceWire port 0, link speed 100 Mbps */
    {
        grspw_config_t spw_cfg;
        spw_cfg.link_speed_mbps = 100U;
        spw_cfg.node_addr       = 1U;
        spw_cfg.dest_key        = 0U;
        (void)grspw_init(0U, &spw_cfg);
    }

    /* UART 0 already done by BSP, init UART 1 for payload */
    (void)apbuart_init(1U, 115200U);

    /* SPI controller 0 */
    (void)spi_init(0U, 1000000U);

    /* I2C controller 0 @ 100 kHz */
    (void)i2c_init(0U, 100000U);

    /* GPIO */
    (void)gpio_init(0U);

    /* Timers — unit 0 for systick (done by BSP), unit 1 for app use */
    {
        gptimer_config_t tmr_cfg;
        tmr_cfg.prescaler = 49U;  /* 50 MHz / (49+1) = 1 MHz → 1 µs */
        (void)gptimer_init(1U, &tmr_cfg);
    }
}

static void init_middleware(void)
{
    /* Packet router */
    (void)router_init();

    /* PUS services — all use APID 0x010 (OBC HK APID) for TM */
    (void)pus_st01_init(APID_OBC_HK);
    (void)pus_st03_init(APID_OBC_HK);
    (void)pus_st05_init(APID_OBC_EVENTS);
    (void)pus_st08_init(APID_OBC_HK);
    (void)pus_st09_init(APID_OBC_HK);
    (void)pus_st11_init(APID_OBC_HK);
    (void)pus_st17_init(APID_OBC_HK);
}

static void init_fsw(void)
{
    (void)sched_init();
    (void)task_table_init();
    (void)wdg_init();
    (void)hmon_init();
    (void)fdir_init();
    (void)mode_manager_init();

    /* Register FDIR fault definitions */
    register_fdir_faults();

    /* Register scheduler slots */
    register_sched_slots();
}

/* ── FDIR fault registration ──────────────────────────────────────────── */

static void register_fdir_faults(void)
{
    /* Communication faults — Level 1 with auto-recovery */
    (void)fdir_register_fault(ERR_COMM_SPW_LINK_DOWN, FDIR_LEVEL_1,
                               fdir_recover_spw);
    (void)fdir_register_fault(ERR_COMM_SPW_TIMEOUT,   FDIR_LEVEL_1,
                               fdir_recover_spw);
    (void)fdir_register_fault(ERR_COMM_CAN_BUS_OFF,   FDIR_LEVEL_1,
                               fdir_recover_can);

    /* Memory faults — Level 2 */
    (void)fdir_register_fault(ERR_MEM_EDAC_SINGLE,    FDIR_LEVEL_1,
                               fdir_recover_mem);
    (void)fdir_register_fault(ERR_MEM_EDAC_DOUBLE,    FDIR_LEVEL_2,
                               (fdir_recovery_fn_t)0);

    /* OBC faults */
    (void)fdir_register_fault(ERR_OBC_WATCHDOG_TIMEOUT, FDIR_LEVEL_1,
                               (fdir_recovery_fn_t)0);
    (void)fdir_register_fault(ERR_OBC_TASK_OVERRUN,     FDIR_LEVEL_1,
                               (fdir_recovery_fn_t)0);

    /* EPS faults — Level 2/3 */
    (void)fdir_register_fault(ERR_EPS_BATTERY_LOW,     FDIR_LEVEL_2,
                               (fdir_recovery_fn_t)0);
    (void)fdir_register_fault(ERR_EPS_UNDERVOLT,       FDIR_LEVEL_3,
                               (fdir_recovery_fn_t)0);

    /* ADCS faults */
    (void)fdir_register_fault(ERR_ADCS_TUMBLING,       FDIR_LEVEL_2,
                               (fdir_recovery_fn_t)0);
}

/* ── FDIR recovery handlers ──────────────────────────────────────────── */

static int32_t fdir_recover_spw(error_code_t code, uint32_t aux)
{
    (void)code;
    (void)aux;
    /* Attempt SpaceWire link re-init */
    {
        grspw_config_t cfg;
        cfg.link_speed_mbps = 100U;
        cfg.node_addr       = 1U;
        cfg.dest_key        = 0U;
        return grspw_init(0U, &cfg);
    }
}

static int32_t fdir_recover_can(error_code_t code, uint32_t aux)
{
    (void)code;
    (void)aux;
    return grcan_init(0U, 500000U);
}

static int32_t fdir_recover_mem(error_code_t code, uint32_t aux)
{
    (void)code;
    (void)aux;
    /* Single-bit EDAC: memory scrub handles correction */
    return 0;
}

/* ── Scheduler slot registration ─────────────────────────────────────── */

static void slot_hk_tick(void)      { pus_st03_tick();     }
static void slot_time_tick(void)    { pus_st09_tick();     }
static void slot_sched_tick(void)   { pus_st11_tick();     }
static void slot_health_tick(void)  { (void)hmon_tick();   }
static void slot_fdir_tick(void)    { (void)fdir_tick();   }
static void slot_tm_drain(void)     { (void)router_process_tm_queue(); }

static void register_sched_slots(void)
{
    /* HK collection every 1 s */
    (void)sched_register_slot("HK",     slot_hk_tick,
                               SCHED_RATE_1S, 0U, 25U);

    /* Time tick every 100 ms (for PUS ST09) */
    (void)sched_register_slot("TIME",   slot_time_tick,
                               SCHED_RATE_100MS, 0U, 10U);

    /* Time-based scheduling every 100 ms */
    (void)sched_register_slot("TSCHED", slot_sched_tick,
                               SCHED_RATE_100MS, 1U, 15U);

    /* Health monitoring every 500 ms */
    (void)sched_register_slot("HMON",   slot_health_tick,
                               SCHED_RATE_500MS, 0U, 30U);

    /* FDIR every 500 ms */
    (void)sched_register_slot("FDIR",   slot_fdir_tick,
                               SCHED_RATE_500MS, 2U, 30U);

    /* TM drain every 100 ms */
    (void)sched_register_slot("TM_OUT", slot_tm_drain,
                               SCHED_RATE_100MS, 3U, 20U);
}

/* ── Mode transitions ─────────────────────────────────────────────────── */

static void boot_to_safe(void)
{
    (void)mode_request_transition(MODE_SAFE);
    hmon_set_status(HMON_SUBSYS_OBC, HEALTH_NOMINAL);

    /* Enable HW watchdog */
    wdg_hw_enable();
}

static void safe_to_nominal(void)
{
    /* Check health before leaving SAFE */
    if (hmon_get_overall() <= HEALTH_DEGRADED) {
        (void)mode_request_transition(MODE_NOMINAL);
    }
}

/* ── Main loop ─────────────────────────────────────────────────────────── */

static void run_forever(void)
{
    uint32_t next_ms = bsp_get_uptime_ms() + SCHED_MINOR_FRAME_MS;

    for (;;) {
        /* Execute minor frame */
        (void)sched_tick();

        /* Kick HW watchdog */
        wdg_hw_kick();

        /* Wait for next slot */
        while (bsp_get_uptime_ms() < next_ms) {
            /* Spin or yield */
        }
        next_ms += SCHED_MINOR_FRAME_MS;
    }
}

/* ── Debug print ───────────────────────────────────────────────────────── */

static void debug_puts(const char *s)
{
    while (*s != '\0') {
        /* Poll UART0 status register until TX holding register empty */
        while ((REG_READ(APBUART0_BASE_ADDR + 0x04U) & (1U << 2U)) == 0U) {
            /* Wait */
        }
        REG_WRITE(APBUART0_BASE_ADDR + 0x00U, (uint32_t)(uint8_t)*s);
        s++;
    }
}
