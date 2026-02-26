/**
 * @file test_watchdog.c
 * @brief Unit tests for watchdog and health monitor modules.
 *
 * Tests software heartbeat registration, timeout detection, health
 * state aggregation, and health vector computation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

/* ── BSP Stubs ─────────────────────────────────────────────────────────── */
static uint32_t stub_uptime_ms = 0U;

uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/*
 * Stub out HW timer writes – watchdog.c writes GPTIMER registers.
 * We define a scratch memory area so the volatile writes don't fault.
 */
static uint32_t hw_timer_scratch[256];

/* Redirect GPTIMER base address for tests.
 * The real hw_config.h defines GPTIMER0_BASE = 0x80000300.
 * We override via macro before including the source. */
#define GPTIMER0_BASE ((uint32_t)(uintptr_t)hw_timer_scratch)

/* We need to provide a minimal hw_config.h shim and then include headers */
/* For simplicity, declare the functions directly and link to source. */

/* ── Minimal forward declarations (match watchdog.h) ──────────────────── */
#define WDG_MAX_MONITORED  16U
#define WDG_HW_TIMEOUT_S    2U
#define WDG_SW_DEFAULT_MS  500U

typedef struct {
    uint8_t  active;
    uint32_t timeout_ms;
    uint32_t last_heartbeat_ms;
} wdg_sw_entry_t;

int32_t  wdg_init(void);
int32_t  wdg_sw_register(uint8_t task_id, uint32_t timeout_ms);
int32_t  wdg_sw_heartbeat(uint8_t task_id);
int32_t  wdg_kick_hw(void);
int32_t  wdg_check_all(uint8_t *expired_id);

/* Health monitor forward declarations */
#define HMON_SUBSYS_MAX  9U

typedef enum {
    HEALTH_UNKNOWN  = 0,
    HEALTH_NOMINAL  = 1,
    HEALTH_DEGRADED = 2,
    HEALTH_FAULTY   = 3,
    HEALTH_OFFLINE  = 4
} health_state_t;

typedef enum {
    SUBSYS_OBC     = 0,
    SUBSYS_EPS     = 1,
    SUBSYS_ADCS    = 2,
    SUBSYS_COMMS   = 3,
    SUBSYS_PAYLOAD = 4,
    SUBSYS_THERMAL = 5,
    SUBSYS_SPW     = 6,
    SUBSYS_CAN     = 7,
    SUBSYS_MEM     = 8
} subsystem_id_t;

int32_t       hmon_init(void);
int32_t       hmon_set(subsystem_id_t id, health_state_t state);
health_state_t hmon_get(subsystem_id_t id);
health_state_t hmon_get_worst(void);
uint32_t       hmon_get_vector(void);
int32_t        hmon_tick(void);

/* ── Test helpers ──────────────────────────────────────────────────────── */
static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)  do { \
    printf("  TEST %-40s ", #name); \
    tests_run++; \
    name(); \
    tests_passed++; \
    printf("[PASS]\n"); \
} while(0)

/* ══════════════════════════════════════════════════════════════════════════
 *  WATCHDOG TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_wdg_init(void)
{
    int32_t ret = wdg_init();
    assert(ret == 0);
}

static void test_sw_register(void)
{
    (void)wdg_init();

    int32_t ret = wdg_sw_register(0U, 500U);
    assert(ret == 0);

    ret = wdg_sw_register(1U, 1000U);
    assert(ret == 0);
}

static void test_sw_heartbeat(void)
{
    (void)wdg_init();
    (void)wdg_sw_register(0U, 500U);

    stub_uptime_ms = 100U;
    int32_t ret = wdg_sw_heartbeat(0U);
    assert(ret == 0);
}

static void test_no_expiry_before_timeout(void)
{
    (void)wdg_init();
    stub_uptime_ms = 0U;
    (void)wdg_sw_register(0U, 500U);
    (void)wdg_sw_heartbeat(0U);

    /* Advance time but stay within timeout */
    stub_uptime_ms = 400U;
    uint8_t expired_id = 0xFFU;
    int32_t ret = wdg_check_all(&expired_id);
    assert(ret == 0);  /* no expiry */
}

static void test_expiry_after_timeout(void)
{
    (void)wdg_init();
    stub_uptime_ms = 0U;
    (void)wdg_sw_register(0U, 500U);
    (void)wdg_sw_heartbeat(0U);

    /* Advance well past timeout */
    stub_uptime_ms = 600U;
    uint8_t expired_id = 0xFFU;
    int32_t ret = wdg_check_all(&expired_id);
    /* ret should be >0 indicating expiry detected, expired_id = 0 */
    assert(ret > 0);
    assert(expired_id == 0U);
}

static void test_hw_kick(void)
{
    (void)wdg_init();
    int32_t ret = wdg_kick_hw();
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  HEALTH MONITOR TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_hmon_init(void)
{
    int32_t ret = hmon_init();
    assert(ret == 0);
}

static void test_hmon_set_get(void)
{
    (void)hmon_init();

    (void)hmon_set(SUBSYS_OBC, HEALTH_NOMINAL);
    assert(hmon_get(SUBSYS_OBC) == HEALTH_NOMINAL);

    (void)hmon_set(SUBSYS_EPS, HEALTH_DEGRADED);
    assert(hmon_get(SUBSYS_EPS) == HEALTH_DEGRADED);
}

static void test_hmon_worst(void)
{
    (void)hmon_init();

    (void)hmon_set(SUBSYS_OBC, HEALTH_NOMINAL);
    (void)hmon_set(SUBSYS_EPS, HEALTH_NOMINAL);
    (void)hmon_set(SUBSYS_ADCS, HEALTH_FAULTY);

    health_state_t worst = hmon_get_worst();
    assert(worst == HEALTH_FAULTY);
}

static void test_hmon_vector(void)
{
    (void)hmon_init();

    /* Set all to NOMINAL (value 1) */
    for (int i = 0; i < (int)HMON_SUBSYS_MAX; i++) {
        (void)hmon_set((subsystem_id_t)i, HEALTH_NOMINAL);
    }

    uint32_t vec = hmon_get_vector();
    /* Each subsystem gets 2 bits, NOMINAL = 0b01
     * So subsys 0 should be bits [1:0] = 01 */
    assert((vec & 0x03U) == 1U);
}

static void test_hmon_tick(void)
{
    (void)hmon_init();
    int32_t ret = hmon_tick();
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== Watchdog & Health Monitor Unit Tests ===\n");

    printf("\n-- Watchdog --\n");
    TEST(test_wdg_init);
    TEST(test_sw_register);
    TEST(test_sw_heartbeat);
    TEST(test_no_expiry_before_timeout);
    TEST(test_expiry_after_timeout);
    TEST(test_hw_kick);

    printf("\n-- Health Monitor --\n");
    TEST(test_hmon_init);
    TEST(test_hmon_set_get);
    TEST(test_hmon_worst);
    TEST(test_hmon_vector);
    TEST(test_hmon_tick);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
