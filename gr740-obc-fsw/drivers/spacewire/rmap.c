/**
 * @file rmap.c
 * @brief RMAP (Remote Memory Access Protocol) target implementation.
 *
 * Processes RMAP read/write commands over SpaceWire, verifies CRC-8,
 * and generates reply packets per ECSS-E-ST-50-52C.
 *
 * @reference ECSS-E-ST-50-52C, RMAP Protocol Specification
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "rmap.h"
#include "grspw.h"
#include "../../bsp/gr740_init.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ======================================================================
 * RMAP Packet Field Offsets (target side)
 * ====================================================================== */
#define RMAP_DEST_ADDR          0U
#define RMAP_PROTOCOL_ID        1U  /**< Protocol ID = 0x01 for RMAP */
#define RMAP_INSTRUCTION        2U
#define RMAP_DEST_KEY           3U
#define RMAP_SRC_ADDR           4U  /* Will be at varying offset */
#define RMAP_TRANSACTION_ID_HI  5U
#define RMAP_TRANSACTION_ID_LO  6U
#define RMAP_EXT_ADDR           7U
#define RMAP_ADDR_BYTE3         8U
#define RMAP_ADDR_BYTE2         9U
#define RMAP_ADDR_BYTE1         10U
#define RMAP_ADDR_BYTE0         11U
#define RMAP_DATA_LEN_HI        12U
#define RMAP_DATA_LEN_MI        13U
#define RMAP_DATA_LEN_LO        14U
#define RMAP_HEADER_CRC         15U

/* RMAP instruction byte bits */
#define RMAP_INST_CMD_BIT       (1U << 6)   /**< 1=command, 0=reply     */
#define RMAP_INST_WRITE_BIT     (1U << 5)   /**< 1=write, 0=read        */
#define RMAP_INST_VERIFY_BIT    (1U << 4)   /**< Verify data             */
#define RMAP_INST_REPLY_BIT     (1U << 3)   /**< Reply requested         */
#define RMAP_INST_INCREMENT_BIT (1U << 2)   /**< Address increment       */

#define RMAP_PROTOCOL_ID_VALUE  0x01U

/* ======================================================================
 * RMAP CRC-8 Lookup Table (polynomial 0x07, per RMAP standard)
 * ====================================================================== */
static const uint8_t rmap_crc_table[256] = {
    0x00, 0x91, 0xE3, 0x72, 0x07, 0x96, 0xE4, 0x75,
    0x0E, 0x9F, 0xED, 0x7C, 0x09, 0x98, 0xEA, 0x7B,
    0x1C, 0x8D, 0xFF, 0x6E, 0x1B, 0x8A, 0xF8, 0x69,
    0x12, 0x83, 0xF1, 0x60, 0x15, 0x84, 0xF6, 0x67,
    0x38, 0xA9, 0xDB, 0x4A, 0x3F, 0xAE, 0xDC, 0x4D,
    0x36, 0xA7, 0xD5, 0x44, 0x31, 0xA0, 0xD2, 0x43,
    0x24, 0xB5, 0xC7, 0x56, 0x23, 0xB2, 0xC0, 0x51,
    0x2A, 0xBB, 0xC9, 0x58, 0x2D, 0xBC, 0xCE, 0x5F,
    0x70, 0xE1, 0x93, 0x02, 0x77, 0xE6, 0x94, 0x05,
    0x7E, 0xEF, 0x9D, 0x0C, 0x79, 0xE8, 0x9A, 0x0B,
    0x6C, 0xFD, 0x8F, 0x1E, 0x6B, 0xFA, 0x88, 0x19,
    0x62, 0xF3, 0x81, 0x10, 0x65, 0xF4, 0x86, 0x17,
    0x48, 0xD9, 0xAB, 0x3A, 0x4F, 0xDE, 0xAC, 0x3D,
    0x46, 0xD7, 0xA5, 0x34, 0x41, 0xD0, 0xA2, 0x33,
    0x54, 0xC5, 0xB7, 0x26, 0x53, 0xC2, 0xB0, 0x21,
    0x5A, 0xCB, 0xB9, 0x28, 0x5D, 0xCC, 0xBE, 0x2F,
    0xE0, 0x71, 0x03, 0x92, 0xE7, 0x76, 0x04, 0x95,
    0xEE, 0x7F, 0x0D, 0x9C, 0xE9, 0x78, 0x0A, 0x9B,
    0xFC, 0x6D, 0x1F, 0x8E, 0xFB, 0x6A, 0x18, 0x89,
    0xF2, 0x63, 0x11, 0x80, 0xF5, 0x64, 0x16, 0x87,
    0xD8, 0x49, 0x3B, 0xAA, 0xDF, 0x4E, 0x3C, 0xAD,
    0xD6, 0x47, 0x35, 0xA4, 0xD1, 0x40, 0x32, 0xA3,
    0xC4, 0x55, 0x27, 0xB6, 0xC3, 0x52, 0x20, 0xB1,
    0xCA, 0x5B, 0x29, 0xB8, 0xCD, 0x5C, 0x2E, 0xBF,
    0x90, 0x01, 0x73, 0xE2, 0x97, 0x06, 0x74, 0xE5,
    0x9E, 0x0F, 0x7D, 0xEC, 0x99, 0x08, 0x7A, 0xEB,
    0x8C, 0x1D, 0x6F, 0xFE, 0x8B, 0x1A, 0x68, 0xF9,
    0x82, 0x13, 0x61, 0xF0, 0x85, 0x14, 0x66, 0xF7,
    0xA8, 0x39, 0x4B, 0xDA, 0xAF, 0x3E, 0x4C, 0xDD,
    0xA6, 0x37, 0x45, 0xD4, 0xA1, 0x30, 0x42, 0xD3,
    0xB4, 0x25, 0x57, 0xC6, 0xB3, 0x22, 0x50, 0xC1,
    0xBA, 0x2B, 0x59, 0xC8, 0xBD, 0x2C, 0x5E, 0xCF
};

/* ======================================================================
 * Private Data
 * ====================================================================== */

static uint8_t  s_rmap_dest_key = 0U;
static uint8_t  s_rmap_spw_port = 0U;
static uint8_t  s_rmap_initialized = 0U;

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Initialize RMAP target.
 * @param[in] spw_port SpW port.
 * @param[in] dest_key RMAP destination key.
 * @return RMAP_OK on success.
 */
int32_t rmap_init(uint8_t spw_port, uint8_t dest_key)
{
    if (spw_port >= SPW_PORT_COUNT) {
        return RMAP_ERR_PARAM;
    }

    s_rmap_spw_port = spw_port;
    s_rmap_dest_key = dest_key;
    s_rmap_initialized = 1U;

    return RMAP_OK;
}

/**
 * @brief Compute RMAP CRC-8.
 * @param[in] data Data bytes.
 * @param[in] len  Length.
 * @return CRC-8 value.
 */
uint8_t rmap_crc8(const uint8_t *data, uint32_t len)
{
    uint8_t crc = 0U;
    uint32_t i;

    if (data == NULL) {
        return 0U;
    }

    for (i = 0U; i < len; i++) {
        crc = rmap_crc_table[crc ^ data[i]];
    }

    return crc;
}

/**
 * @brief Verify RMAP CRC on a packet.
 * @param[in] packet Packet data.
 * @param[in] len    Packet length (including CRC byte).
 * @return RMAP_OK if valid, RMAP_ERR_CRC if invalid.
 */
int32_t rmap_verify_crc(const uint8_t *packet, uint32_t len)
{
    uint8_t crc;

    if ((packet == NULL) || (len < 2U)) {
        return RMAP_ERR_PARAM;
    }

    /* CRC covers all bytes; result should be 0 if CRC byte included */
    crc = rmap_crc8(packet, len);

    return (crc == 0U) ? RMAP_OK : RMAP_ERR_CRC;
}

/**
 * @brief Process an incoming RMAP command and generate reply.
 * @param[in]  packet     RMAP command packet.
 * @param[in]  pkt_len    Command packet length.
 * @param[out] reply      Reply buffer.
 * @param[out] reply_len  Reply length.
 * @return RMAP_OK on success.
 */
int32_t rmap_process_command(const uint8_t *packet, uint32_t pkt_len,
                             uint8_t *reply, uint32_t *reply_len)
{
    uint8_t  instruction;
    uint8_t  key;
    uint32_t address;
    uint32_t data_len;
    uint8_t  header_crc;
    uint32_t reply_idx;

    if ((packet == NULL) || (reply == NULL) || (reply_len == NULL)) {
        return RMAP_ERR_PARAM;
    }
    if (pkt_len < (RMAP_HEADER_CRC + 1U)) {
        return RMAP_ERR_PARAM;
    }
    if (s_rmap_initialized == 0U) {
        return RMAP_ERR_COMMAND;
    }

    /* Verify protocol ID */
    if (packet[RMAP_PROTOCOL_ID] != RMAP_PROTOCOL_ID_VALUE) {
        return RMAP_ERR_COMMAND;
    }

    /* Extract fields */
    instruction = packet[RMAP_INSTRUCTION];
    key = packet[RMAP_DEST_KEY];

    /* Verify destination key */
    if (key != s_rmap_dest_key) {
        return RMAP_ERR_COMMAND;
    }

    /* Verify this is a command (not a reply) */
    if ((instruction & RMAP_INST_CMD_BIT) == 0U) {
        return RMAP_ERR_COMMAND;
    }

    /* Verify header CRC */
    header_crc = rmap_crc8(packet, RMAP_HEADER_CRC);
    if (header_crc != packet[RMAP_HEADER_CRC]) {
        return RMAP_ERR_CRC;
    }

    /* Extract memory address */
    address = ((uint32_t)packet[RMAP_ADDR_BYTE3] << 24) |
              ((uint32_t)packet[RMAP_ADDR_BYTE2] << 16) |
              ((uint32_t)packet[RMAP_ADDR_BYTE1] << 8) |
              ((uint32_t)packet[RMAP_ADDR_BYTE0]);

    /* Extract data length */
    data_len = ((uint32_t)packet[RMAP_DATA_LEN_HI] << 16) |
               ((uint32_t)packet[RMAP_DATA_LEN_MI] << 8) |
               ((uint32_t)packet[RMAP_DATA_LEN_LO]);

    reply_idx = 0U;

    if ((instruction & RMAP_INST_WRITE_BIT) != 0U) {
        /* WRITE command */
        /* Verify data CRC */
        if ((RMAP_HEADER_CRC + 1U + data_len + 1U) > pkt_len) {
            return RMAP_ERR_PARAM;
        }

        const uint8_t *write_data = &packet[RMAP_HEADER_CRC + 1U];
        uint8_t data_crc = rmap_crc8(write_data, data_len);
        if (data_crc != write_data[data_len]) {
            return RMAP_ERR_CRC;
        }

        /* Perform memory write */
        volatile uint8_t *mem_ptr = (volatile uint8_t *)address;
        uint32_t i;
        for (i = 0U; i < data_len; i++) {
            mem_ptr[i] = write_data[i];
        }

        /* Generate write reply if requested */
        if ((instruction & RMAP_INST_REPLY_BIT) != 0U) {
            reply[reply_idx++] = packet[RMAP_SRC_ADDR];   /* Initiator addr */
            reply[reply_idx++] = RMAP_PROTOCOL_ID_VALUE;
            reply[reply_idx++] = instruction & (uint8_t)(~RMAP_INST_CMD_BIT);
            reply[reply_idx++] = 0x00U; /* Status: success */
            reply[reply_idx++] = packet[RMAP_DEST_ADDR]; /* Target addr */
            reply[reply_idx++] = packet[RMAP_TRANSACTION_ID_HI];
            reply[reply_idx++] = packet[RMAP_TRANSACTION_ID_LO];
            reply[reply_idx++] = rmap_crc8(reply, reply_idx);
        }
    } else {
        /* READ command */
        /* Generate read reply */
        reply[reply_idx++] = packet[RMAP_SRC_ADDR];
        reply[reply_idx++] = RMAP_PROTOCOL_ID_VALUE;
        reply[reply_idx++] = instruction & (uint8_t)(~RMAP_INST_CMD_BIT);
        reply[reply_idx++] = 0x00U; /* Status: success */
        reply[reply_idx++] = packet[RMAP_DEST_ADDR];
        reply[reply_idx++] = packet[RMAP_TRANSACTION_ID_HI];
        reply[reply_idx++] = packet[RMAP_TRANSACTION_ID_LO];
        reply[reply_idx++] = 0x00U; /* Reserved */
        reply[reply_idx++] = (uint8_t)(data_len >> 16);
        reply[reply_idx++] = (uint8_t)(data_len >> 8);
        reply[reply_idx++] = (uint8_t)(data_len);
        reply[reply_idx++] = rmap_crc8(reply, reply_idx); /* Header CRC */

        /* Copy data from memory */
        volatile uint8_t *mem_ptr = (volatile uint8_t *)address;
        uint32_t i;
        for (i = 0U; i < data_len; i++) {
            reply[reply_idx++] = mem_ptr[i];
        }

        /* Data CRC */
        reply[reply_idx] = rmap_crc8(&reply[12], data_len);
        reply_idx++;
    }

    *reply_len = reply_idx;

    return RMAP_OK;
}
