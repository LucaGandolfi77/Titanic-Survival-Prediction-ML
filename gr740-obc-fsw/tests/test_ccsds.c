/**
 * @file test_ccsds.c
 * @brief Unit tests for CCSDS Space Packet, TM Framing, TC Framing.
 *
 * Simple assert-based tests for host-side verification.
 * Compile: gcc -I../../ -o test_ccsds test_ccsds.c \
 *          ../../middleware/ccsds/space_packet.c \
 *          ../../middleware/ccsds/tm_framing.c \
 *          ../../middleware/ccsds/tc_framing.c
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

/* Stubs for BSP dependencies */
uint32_t bsp_get_uptime_ms(void) { return 12345U; }

#include "../../middleware/ccsds/space_packet.h"
#include "../../middleware/ccsds/tm_framing.h"
#include "../../middleware/ccsds/tc_framing.h"

/* ── Helpers ───────────────────────────────────────────────────────────── */
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
 *  SPACE PACKET TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_crc16_known_vector(void)
{
    /* CCSDS CRC-16 test: "123456789" → 0x29B1 */
    const uint8_t data[] = "123456789";
    uint16_t crc = ccsds_crc16(data, 9U);
    assert(crc == 0x29B1U);
}

static void test_packet_init_tm(void)
{
    ccsds_packet_t pkt;
    int32_t ret = ccsds_init_packet(&pkt, CCSDS_TYPE_TM, 0x010,
                                     CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    assert(ret == CCSDS_OK);

    /* Verify packet ID: version=000, type=0(TM), shdr=1, APID=0x010 */
    uint16_t pkt_id = pkt.header.pkt_id;
    assert((pkt_id & 0xE000U) == 0x0000U);  /* version 000 */
    assert((pkt_id & 0x1000U) == 0x0000U);  /* TM type=0 */
    assert((pkt_id & 0x0800U) == 0x0800U);  /* sec hdr present */
    assert((pkt_id & 0x07FFU) == 0x0010U);  /* APID */
}

static void test_packet_serialize_deserialize(void)
{
    ccsds_packet_t pkt, pkt2;
    uint8_t data[4] = { 0xDE, 0xAD, 0xBE, 0xEF };
    uint8_t buf[256];
    uint16_t ser_len;
    int32_t ret;

    ret = ccsds_init_packet(&pkt, CCSDS_TYPE_TC, 0x020,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    assert(ret == CCSDS_OK);

    ret = ccsds_set_data(&pkt, data, 4U);
    assert(ret == CCSDS_OK);

    ret = ccsds_serialize(&pkt, buf, 256U, &ser_len);
    assert(ret == CCSDS_OK);
    assert(ser_len > 6U);

    ret = ccsds_deserialize(&pkt2, buf, ser_len);
    assert(ret == CCSDS_OK);
    assert(pkt2.header.pkt_id == pkt.header.pkt_id);
    assert(pkt2.data_len == pkt.data_len);
}

static void test_seq_counter_increment(void)
{
    uint16_t s1 = ccsds_next_seq_count(0x030);
    uint16_t s2 = ccsds_next_seq_count(0x030);
    assert(s2 == (uint16_t)(s1 + 1U));
}

/* ══════════════════════════════════════════════════════════════════════════
 *  TM FRAMING TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_tm_frame_init(void)
{
    int32_t ret = tm_frame_init(0x1AU);  /* SCID */
    assert(ret == TM_FRAME_OK);
}

static void test_tm_frame_assemble(void)
{
    tm_frame_t frame;
    uint8_t data[128];
    int32_t ret;

    memset(data, 0xAA, 128);
    (void)tm_frame_init(0x1AU);

    ret = tm_frame_assemble(0U, data, 128U, &frame);
    assert(ret == TM_FRAME_OK);
    assert(frame.data_len > 0U);
}

static void test_tm_idle_frame(void)
{
    tm_frame_t frame;
    int32_t ret;

    (void)tm_frame_init(0x1AU);
    ret = tm_frame_idle(&frame);
    assert(ret == TM_FRAME_OK);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  TC FRAMING TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_tc_frame_init(void)
{
    int32_t ret = tc_frame_init(0x1AU);
    assert(ret == TC_FRAME_OK);
}

static void test_tc_frame_assemble(void)
{
    tc_frame_t frame;
    uint8_t data[32];
    int32_t ret;

    memset(data, 0x55, 32);
    (void)tc_frame_init(0x1AU);

    ret = tc_frame_assemble(0U, 0U, data, 32U, &frame);
    assert(ret == TC_FRAME_OK);
    assert(frame.frame_len > 0U);
}

static void test_cltu_encode_decode(void)
{
    tc_frame_t frame;
    uint8_t data[16];
    uint8_t cltu_buf[256];
    uint16_t cltu_len;
    uint8_t decoded[256];
    uint16_t decoded_len;
    int32_t ret;

    memset(data, 0x42, 16);
    (void)tc_frame_init(0x1AU);
    ret = tc_frame_assemble(0U, 0U, data, 16U, &frame);
    assert(ret == TC_FRAME_OK);

    ret = tc_cltu_encode(&frame, cltu_buf, 256U, &cltu_len);
    assert(ret == TC_FRAME_OK);
    assert(cltu_len > 0U);

    /* Verify start sequence */
    assert(cltu_buf[0] == 0xEBU);
    assert(cltu_buf[1] == 0x90U);

    ret = tc_cltu_decode(cltu_buf, cltu_len, decoded, 256U, &decoded_len);
    assert(ret == TC_FRAME_OK);
    assert(decoded_len == frame.frame_len);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== CCSDS Unit Tests ===\n");

    printf("\n-- Space Packet --\n");
    TEST(test_crc16_known_vector);
    TEST(test_packet_init_tm);
    TEST(test_packet_serialize_deserialize);
    TEST(test_seq_counter_increment);

    printf("\n-- TM Framing --\n");
    TEST(test_tm_frame_init);
    TEST(test_tm_frame_assemble);
    TEST(test_tm_idle_frame);

    printf("\n-- TC Framing --\n");
    TEST(test_tc_frame_init);
    TEST(test_tc_frame_assemble);
    TEST(test_cltu_encode_decode);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
