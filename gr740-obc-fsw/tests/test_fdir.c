/**
 * @file test_fdir.c
 * @brief Unit tests for FDIR module — fault registration, reporting,
 *        escalation, recovery callbacks, and history.
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

/* ── BSP stubs ─────────────────────────────────────────────────────────── */
static uint32_t stub_uptime_ms = 0U;
uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/* ── ST05 event stub ───────────────────────────────────────────────────── */
static uint16_t last_event_id       = 0U;
static uint8_t  last_event_severity = 0U;
static uint32_t event_count         = 0U;

void pus_st05_raise_event(uint8_t severity, uint16_t event_id,
                          const uint8_t *aux, uint16_t aux_len)
{
    (void)aux;
    (void)aux_len;
    last_event_severity = severity;
    last_event_id       = event_id;
    event_count++;
}

/* ── Mode manager stub for safe mode ───────────────────────────────────── */
static int safe_mode_requests = 0;

int32_t mode_request_safe(void)
{
    safe_mode_requests++;
    return 0;
}

/* ── Include FDIR types (simplified inline declarations) ──────────────── */

#define FDIR_MAX_FAULTS   64U
#define FDIR_HISTORY_DEPTH 32U
#define FDIR_RETRY_LIMIT   3U

typedef enum {
    FDIR_LEVEL_1 = 1,  /* Local retry       */
    FDIR_LEVEL_2 = 2,  /* Isolate subsystem */
    FDIR_LEVEL_3 = 3   /* System-level      */
} fdir_level_t;

typedef int32_t (*fdir_recovery_fn)(uint16_t fault_id);

typedef struct {
    uint16_t        fault_id;
    uint32_t        timestamp_ms;
    int32_t         recovery_result;
} fdir_history_entry_t;

int32_t  fdir_init(void);
int32_t  fdir_register_fault(uint16_t fault_id, fdir_level_t level,
                              fdir_recovery_fn recovery);
int32_t  fdir_report_fault(uint16_t fault_id);
uint32_t fdir_get_history_count(void);
int32_t  fdir_get_history(uint32_t index, fdir_history_entry_t *entry);
int32_t  fdir_clear_fault(uint16_t fault_id);
int32_t  fdir_clear_all(void);

/* ── Test infrastructure ──────────────────────────────────────────────── */
static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)  do { \
    printf("  TEST %-40s ", #name); \
    tests_run++; \
    name(); \
    tests_passed++; \
    printf("[PASS]\n"); \
} while(0)

/* ── Recovery callback tracker ────────────────────────────────────────── */
static int recovery_call_count = 0;
static uint16_t recovery_last_fault = 0U;

static int32_t mock_recovery(uint16_t fault_id)
{
    recovery_call_count++;
    recovery_last_fault = fault_id;
    return 0;  /* success */
}

static int32_t mock_recovery_fail(uint16_t fault_id)
{
    recovery_call_count++;
    recovery_last_fault = fault_id;
    return -1;  /* failure — triggers escalation */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_fdir_init(void)
{
    int32_t ret = fdir_init();
    assert(ret == 0);
}

static void test_register_fault(void)
{
    (void)fdir_init();
    recovery_call_count = 0;

    int32_t ret = fdir_register_fault(0x0100U, FDIR_LEVEL_1, mock_recovery);
    assert(ret == 0);
}

static void test_report_and_recover(void)
{
    (void)fdir_init();
    recovery_call_count = 0;
    stub_uptime_ms = 1000U;

    (void)fdir_register_fault(0x0200U, FDIR_LEVEL_1, mock_recovery);

    int32_t ret = fdir_report_fault(0x0200U);
    assert(ret == 0);

    /* Recovery callback should have been invoked */
    assert(recovery_call_count == 1);
    assert(recovery_last_fault == 0x0200U);
}

static void test_history_recorded(void)
{
    (void)fdir_init();
    recovery_call_count = 0;
    stub_uptime_ms = 2000U;

    (void)fdir_register_fault(0x0300U, FDIR_LEVEL_1, mock_recovery);
    (void)fdir_report_fault(0x0300U);

    uint32_t count = fdir_get_history_count();
    assert(count >= 1U);

    fdir_history_entry_t entry;
    int32_t ret = fdir_get_history(0U, &entry);
    assert(ret == 0);
    assert(entry.fault_id == 0x0300U);
    assert(entry.timestamp_ms == 2000U);
    assert(entry.recovery_result == 0);  /* mock_recovery returns 0 */
}

static void test_escalation_on_repeated_failure(void)
{
    (void)fdir_init();
    recovery_call_count = 0;
    event_count = 0U;
    safe_mode_requests = 0;
    stub_uptime_ms = 3000U;

    /* Register as L1 with a recovery that always fails */
    (void)fdir_register_fault(0x0400U, FDIR_LEVEL_1, mock_recovery_fail);

    /* Report FDIR_RETRY_LIMIT (3) times to exhaust L1 retries */
    for (uint32_t i = 0U; i < FDIR_RETRY_LIMIT; i++) {
        stub_uptime_ms += 100U;
        (void)fdir_report_fault(0x0400U);
    }

    /* After 3 failures the fault should have escalated.
     * Event(s) should have been raised via pus_st05_raise_event. */
    assert(event_count > 0U);
}

static void test_clear_fault(void)
{
    (void)fdir_init();
    recovery_call_count = 0;

    (void)fdir_register_fault(0x0500U, FDIR_LEVEL_1, mock_recovery);
    (void)fdir_report_fault(0x0500U);

    int32_t ret = fdir_clear_fault(0x0500U);
    assert(ret == 0);
}

static void test_clear_all(void)
{
    (void)fdir_init();
    int32_t ret = fdir_clear_all();
    assert(ret == 0);
    assert(fdir_get_history_count() == 0U);
}

static void test_unregistered_fault_rejected(void)
{
    (void)fdir_init();

    /* Report a fault that was never registered */
    int32_t ret = fdir_report_fault(0xFFFFU);
    assert(ret != 0);
}

static void test_max_faults(void)
{
    (void)fdir_init();

    /* Register up to FDIR_MAX_FAULTS entries */
    for (uint16_t i = 0U; i < (uint16_t)FDIR_MAX_FAULTS; i++) {
        int32_t ret = fdir_register_fault(i, FDIR_LEVEL_1, mock_recovery);
        assert(ret == 0);
    }

    /* One more should fail */
    int32_t ret = fdir_register_fault(0xFFFEU, FDIR_LEVEL_1, mock_recovery);
    assert(ret != 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== FDIR Unit Tests ===\n\n");

    TEST(test_fdir_init);
    TEST(test_register_fault);
    TEST(test_report_and_recover);
    TEST(test_history_recorded);
    TEST(test_escalation_on_repeated_failure);
    TEST(test_clear_fault);
    TEST(test_clear_all);
    TEST(test_unregistered_fault_rejected);
    TEST(test_max_faults);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
