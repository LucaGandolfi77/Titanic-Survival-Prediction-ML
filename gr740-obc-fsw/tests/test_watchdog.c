/**
 * @file test_watchdog.c
 * @brief Unit tests for watchdog and health monitor modules.
 *
 * Tests software heartbeat registration, timeout detection, health
 * state aggregation, and health vector computation.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

/* BSP stub: control time returned to modules */
static uint32_t stub_uptime_ms = 0U;
uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/* Provide a small scratch region and override GPTIMER base for host build */
static uint32_t hw_timer_scratch[256];
#define GPTIMER0_BASE ((uint32_t)(uintptr_t)hw_timer_scratch)

#include "fsw/watchdog/watchdog.h"
#include "fsw/watchdog/health_monitor.h"

/* Test harness helpers */
static int tests_run = 0;
static int tests_passed = 0;
#define TEST(name) do { \
    printf("  TEST %-40s ", #name); \
    tests_run++; name(); tests_passed++; printf("[PASS]\n"); \
} while(0)

static void test_wdg_init(void)
{
    int32_t r = wdg_init();
    assert(r == WDG_OK);
}

static void test_register_task(void)
{
    (void)wdg_init();
    assert(wdg_register_task(0U, 500U, "t0") == WDG_OK);
    assert(wdg_register_task(1U, 1000U, "t1") == WDG_OK);
}

static void test_heartbeat(void)
{
    (void)wdg_init(); (void)wdg_register_task(0U, 500U, "t0");
    stub_uptime_ms = 100U;
    assert(wdg_heartbeat(0U) == WDG_OK);
}

static void test_no_expiry_before_timeout(void)
{
    (void)wdg_init(); (void)wdg_register_task(0U, 500U, "t0");
    (void)wdg_heartbeat(0U);
    stub_uptime_ms = 400U;
    uint32_t expired = 0xFFFFFFFFU;
    assert(wdg_check_all(&expired) == 0);
}

static void test_expiry_after_timeout(void)
{
    (void)wdg_init(); (void)wdg_register_task(0U, 500U, "t0");
    (void)wdg_heartbeat(0U);
    stub_uptime_ms = 600U;
    uint32_t expired = 0xFFFFFFFFU;
    assert(wdg_check_all(&expired) > 0);
    assert(expired == 0U);
}

static void test_hw_kick(void)
{
    (void)wdg_init();
    /* Should not crash or return error on host */
    wdg_hw_kick();
}

/* Health monitor tests */
static void test_hmon_init(void)
{
    assert(hmon_init() == HMON_OK);
}

static void test_hmon_set_get(void)
{
    (void)hmon_init();
    assert(hmon_set_status(HMON_SUBSYS_OBC, HEALTH_NOMINAL) == HMON_OK);
    health_status_t s = HEALTH_UNKNOWN;
    assert(hmon_get_status(HMON_SUBSYS_OBC, &s) == HMON_OK);
    assert(s == HEALTH_NOMINAL);
}

static void test_hmon_overall(void)
{
    (void)hmon_init();
    (void)hmon_set_status(HMON_SUBSYS_OBC, HEALTH_NOMINAL);
    (void)hmon_set_status(HMON_SUBSYS_ADCS, HEALTH_FAULTY);
    assert(hmon_get_overall() == HEALTH_FAULTY);
}

static void test_hmon_vector(void)
{
    (void)hmon_init();
    for (int i = 0; i < (int)HMON_SUBSYS_COUNT; i++) {
        (void)hmon_set_status((hmon_subsys_t)i, HEALTH_NOMINAL);
    }
    uint32_t v = hmon_get_vector();
    assert((v & 0x03U) == 1U);
}

int main(void)
{
    printf("=== Watchdog & Health Monitor Unit Tests ===\n");
    TEST(test_wdg_init);
    TEST(test_register_task);
    TEST(test_heartbeat);
    TEST(test_no_expiry_before_timeout);
    TEST(test_expiry_after_timeout);
    TEST(test_hw_kick);

    TEST(test_hmon_init);
    TEST(test_hmon_set_get);
    TEST(test_hmon_overall);
    TEST(test_hmon_vector);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
