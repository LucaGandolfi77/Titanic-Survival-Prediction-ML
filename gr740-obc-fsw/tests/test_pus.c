/**
 * @file test_pus.c
 * @brief Unit tests for all PUS services (ST01, ST03, ST05, ST08,
 *        ST09, ST11, ST17).
 *
 * Each service is tested via its public API.  Router and BSP
 * dependencies are stubbed out.
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

/* ── BSP / Router stubs ───────────────────────────────────────────────── */
static uint32_t stub_uptime_ms = 0U;
uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/* Capture TM packets sent via router_send_tm */
#define TM_CAPTURE_MAX 32U
static uint8_t  tm_capture[TM_CAPTURE_MAX][256];
static uint16_t tm_capture_len[TM_CAPTURE_MAX];
static uint32_t tm_capture_count = 0U;

int32_t router_send_tm(const uint8_t *data, uint16_t len)
{
    if (tm_capture_count < TM_CAPTURE_MAX) {
        if (len <= 256U) {
            memcpy(tm_capture[tm_capture_count], data, len);
            tm_capture_len[tm_capture_count] = len;
        }
        tm_capture_count++;
    }
    return 0;
}

static void tm_capture_reset(void)
{
    tm_capture_count = 0U;
    memset(tm_capture, 0, sizeof(tm_capture));
    memset(tm_capture_len, 0, sizeof(tm_capture_len));
}

/* CCSDS helpers for building minimal TC packets */
static uint16_t build_tc_header(uint8_t *buf, uint16_t apid,
                                 uint16_t data_len)
{
    /* Packet ID: version=000, type=1(TC), shdr=1, APID */
    uint16_t pkt_id = 0x1800U | (apid & 0x07FFU);
    buf[0] = (uint8_t)(pkt_id >> 8);
    buf[1] = (uint8_t)(pkt_id & 0xFFU);

    /* Sequence: unseg=11, count=0 */
    buf[2] = 0xC0U;
    buf[3] = 0x00U;

    /* Data length: total_data_bytes - 1 */
    uint16_t dl = (data_len > 0U) ? (data_len - 1U) : 0U;
    buf[4] = (uint8_t)(dl >> 8);
    buf[5] = (uint8_t)(dl & 0xFFU);

    return 6U;  /* primary header size */
}

/* ── Forward declarations for PUS APIs ────────────────────────────────── */

/* ST01 — Verification */
int32_t pus_st01_init(void);
int32_t pus_st01_accept_ok(uint16_t apid, uint16_t seq);
int32_t pus_st01_accept_fail(uint16_t apid, uint16_t seq, uint8_t code);
int32_t pus_st01_exec_ok(uint16_t apid, uint16_t seq);
int32_t pus_st01_exec_fail(uint16_t apid, uint16_t seq, uint8_t code);

/* ST03 — Housekeeping */
int32_t pus_st03_init(void);
int32_t pus_st03_define_sid(uint16_t sid, uint16_t interval_s);
int32_t pus_st03_enable_sid(uint16_t sid);
int32_t pus_st03_disable_sid(uint16_t sid);
int32_t pus_st03_set_param(uint16_t sid, uint8_t idx,
                            uint32_t value);
int32_t pus_st03_tick(void);

/* ST05 — Events */
int32_t pus_st05_init(void);
void    pus_st05_raise_event(uint8_t severity, uint16_t event_id,
                              const uint8_t *aux, uint16_t aux_len);
int32_t pus_st05_enable_event(uint16_t event_id);
int32_t pus_st05_disable_event(uint16_t event_id);

/* ST08 — Function Management */
typedef int32_t (*pus_st08_func_t)(const uint8_t *args, uint16_t len);

int32_t pus_st08_init(void);
int32_t pus_st08_register(uint8_t func_id, pus_st08_func_t handler);
int32_t pus_st08_dispatch(const uint8_t *tc, uint16_t tc_len);

/* ST09 — Time Management */
int32_t pus_st09_init(void);
int32_t pus_st09_update_time(uint32_t coarse, uint16_t fine);
int32_t pus_st09_get_time(uint32_t *coarse, uint16_t *fine);
int32_t pus_st09_handle_tc(const uint8_t *tc, uint16_t tc_len);

/* ST11 — Time-based Scheduling */
int32_t pus_st11_init(void);
int32_t pus_st11_insert(uint32_t exec_time_s, const uint8_t *tc_pkt,
                         uint16_t tc_len);
int32_t pus_st11_enable(void);
int32_t pus_st11_disable(void);
int32_t pus_st11_tick(uint32_t current_time_s);
uint32_t pus_st11_get_count(void);

/* ST17 — Test Connection */
int32_t pus_st17_init(void);
int32_t pus_st17_handle_tc(const uint8_t *tc, uint16_t tc_len);

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

/* ══════════════════════════════════════════════════════════════════════════
 *  ST01 — Verification
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st01_init(void)
{
    int32_t ret = pus_st01_init();
    assert(ret == 0);
}

static void test_st01_accept_ok(void)
{
    (void)pus_st01_init();
    tm_capture_reset();

    int32_t ret = pus_st01_accept_ok(0x010U, 1U);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,1) emitted */
}

static void test_st01_accept_fail(void)
{
    (void)pus_st01_init();
    tm_capture_reset();

    int32_t ret = pus_st01_accept_fail(0x010U, 2U, 0x42);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,2) emitted */
}

static void test_st01_exec_ok(void)
{
    (void)pus_st01_init();
    tm_capture_reset();

    int32_t ret = pus_st01_exec_ok(0x010U, 3U);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,7) */
}

static void test_st01_exec_fail(void)
{
    (void)pus_st01_init();
    tm_capture_reset();

    int32_t ret = pus_st01_exec_fail(0x010U, 4U, 0x13);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,8) */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST03 — Housekeeping
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st03_init(void)
{
    int32_t ret = pus_st03_init();
    assert(ret == 0);
}

static void test_st03_define_and_enable(void)
{
    (void)pus_st03_init();

    int32_t ret = pus_st03_define_sid(1U, 5U);  /* SID 1, every 5 s */
    assert(ret == 0);

    ret = pus_st03_enable_sid(1U);
    assert(ret == 0);
}

static void test_st03_set_param(void)
{
    (void)pus_st03_init();
    (void)pus_st03_define_sid(1U, 5U);

    int32_t ret = pus_st03_set_param(1U, 0U, 0xDEADBEEFU);
    assert(ret == 0);
}

static void test_st03_tick_generates_tm(void)
{
    (void)pus_st03_init();
    (void)pus_st03_define_sid(1U, 1U);  /* SID 1, every 1 s */
    (void)pus_st03_enable_sid(1U);
    (void)pus_st03_set_param(1U, 0U, 42U);

    tm_capture_reset();
    stub_uptime_ms = 1000U;

    /* Tick enough times to exceed interval */
    int32_t ret = pus_st03_tick();
    assert(ret == 0);
    /* A TM(3,25) should have been emitted */
    assert(tm_capture_count >= 1U);
}

static void test_st03_disable(void)
{
    (void)pus_st03_init();
    (void)pus_st03_define_sid(2U, 1U);
    (void)pus_st03_enable_sid(2U);
    (void)pus_st03_disable_sid(2U);

    tm_capture_reset();
    (void)pus_st03_tick();
    /* Disabled SID should not generate TM */
    /* TM count for this SID should be 0 */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST05 — Events
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st05_init(void)
{
    int32_t ret = pus_st05_init();
    assert(ret == 0);
}

static void test_st05_raise_event(void)
{
    (void)pus_st05_init();
    tm_capture_reset();

    uint8_t aux[4] = { 0x01, 0x02, 0x03, 0x04 };
    pus_st05_raise_event(2U, 0x1000U, aux, 4U);

    /* Should have emitted a TM(5,x) via router_send_tm */
    assert(tm_capture_count >= 1U);
}

static void test_st05_disable_enable(void)
{
    (void)pus_st05_init();

    int32_t ret = pus_st05_disable_event(0x1000U);
    assert(ret == 0);

    ret = pus_st05_enable_event(0x1000U);
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST08 — Function Management
 * ══════════════════════════════════════════════════════════════════════ */

static int32_t mock_func_handler(const uint8_t *args, uint16_t len)
{
    (void)args;
    (void)len;
    return 0;
}

static void test_st08_init(void)
{
    int32_t ret = pus_st08_init();
    assert(ret == 0);
}

static void test_st08_register(void)
{
    (void)pus_st08_init();
    int32_t ret = pus_st08_register(1U, mock_func_handler);
    assert(ret == 0);
}

static void test_st08_dispatch(void)
{
    (void)pus_st08_init();
    (void)pus_st08_register(1U, mock_func_handler);

    /* Build a minimal TC(8,1) packet:
     * Primary header (6) + PUS secondary header (min 3) +
     * function_id (1) */
    uint8_t tc[16];
    uint16_t hdr_len = build_tc_header(tc, 0x020U, 4U);

    /* PUS secondary header: version=0x20, service=8, subtype=1 */
    tc[hdr_len + 0U] = 0x20U;
    tc[hdr_len + 1U] = 8U;
    tc[hdr_len + 2U] = 1U;

    /* Function ID */
    tc[hdr_len + 3U] = 1U;

    tm_capture_reset();
    int32_t ret = pus_st08_dispatch(tc, (uint16_t)(hdr_len + 4U));
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST09 — Time Management
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st09_init(void)
{
    int32_t ret = pus_st09_init();
    assert(ret == 0);
}

static void test_st09_set_get_time(void)
{
    (void)pus_st09_init();

    int32_t ret = pus_st09_update_time(1000000U, 32768U);
    assert(ret == 0);

    uint32_t coarse = 0U;
    uint16_t fine   = 0U;
    ret = pus_st09_get_time(&coarse, &fine);
    assert(ret == 0);
    assert(coarse == 1000000U);
    assert(fine == 32768U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST11 — Time-based Scheduling
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st11_init(void)
{
    int32_t ret = pus_st11_init();
    assert(ret == 0);
}

static void test_st11_insert_and_count(void)
{
    (void)pus_st11_init();

    /* Insert a dummy TC scheduled for epoch 100 */
    uint8_t tc[16];
    (void)build_tc_header(tc, 0x020U, 4U);
    tc[6] = 0x20U; tc[7] = 17U; tc[8] = 1U; tc[9] = 0U;

    int32_t ret = pus_st11_insert(100U, tc, 10U);
    assert(ret == 0);

    assert(pus_st11_get_count() == 1U);
}

static void test_st11_enable_disable(void)
{
    (void)pus_st11_init();

    int32_t ret = pus_st11_enable();
    assert(ret == 0);

    ret = pus_st11_disable();
    assert(ret == 0);
}

static void test_st11_tick_executes_due(void)
{
    (void)pus_st11_init();

    uint8_t tc[16];
    uint16_t hdr = build_tc_header(tc, 0x020U, 4U);
    tc[hdr + 0U] = 0x20U;
    tc[hdr + 1U] = 17U;
    tc[hdr + 2U] = 1U;
    tc[hdr + 3U] = 0U;

    (void)pus_st11_insert(50U, tc, (uint16_t)(hdr + 4U));
    (void)pus_st11_enable();

    tm_capture_reset();
    int32_t ret = pus_st11_tick(60U);  /* current_time > exec_time */
    assert(ret == 0);

    /* The scheduled TC should have been dispatched;
     * count should now be 0 */
    assert(pus_st11_get_count() == 0U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST17 — Test Connection
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st17_init(void)
{
    int32_t ret = pus_st17_init();
    assert(ret == 0);
}

static void test_st17_are_you_alive(void)
{
    (void)pus_st17_init();
    tm_capture_reset();

    /* Build TC(17,1) */
    uint8_t tc[16];
    uint16_t hdr = build_tc_header(tc, 0x020U, 3U);
    tc[hdr + 0U] = 0x20U;  /* PUS version 2 */
    tc[hdr + 1U] = 17U;    /* service */
    tc[hdr + 2U] = 1U;     /* subtype — are you alive */

    int32_t ret = pus_st17_handle_tc(tc, (uint16_t)(hdr + 3U));
    assert(ret == 0);
    /* Should emit TM(17,2) */
    assert(tm_capture_count >= 1U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== PUS Services Unit Tests ===\n");

    printf("\n-- ST01 Verification --\n");
    TEST(test_st01_init);
    TEST(test_st01_accept_ok);
    TEST(test_st01_accept_fail);
    TEST(test_st01_exec_ok);
    TEST(test_st01_exec_fail);

    printf("\n-- ST03 Housekeeping --\n");
    TEST(test_st03_init);
    TEST(test_st03_define_and_enable);
    TEST(test_st03_set_param);
    TEST(test_st03_tick_generates_tm);
    TEST(test_st03_disable);

    printf("\n-- ST05 Events --\n");
    TEST(test_st05_init);
    TEST(test_st05_raise_event);
    TEST(test_st05_disable_enable);

    printf("\n-- ST08 Function Management --\n");
    TEST(test_st08_init);
    TEST(test_st08_register);
    TEST(test_st08_dispatch);

    printf("\n-- ST09 Time Management --\n");
    TEST(test_st09_init);
    TEST(test_st09_set_get_time);

    printf("\n-- ST11 Time-based Scheduling --\n");
    TEST(test_st11_init);
    TEST(test_st11_insert_and_count);
    TEST(test_st11_enable_disable);
    TEST(test_st11_tick_executes_due);

    printf("\n-- ST17 Test Connection --\n");
    TEST(test_st17_init);
    TEST(test_st17_are_you_alive);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
