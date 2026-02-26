/**
 * @file rmap.h
 * @brief RMAP (Remote Memory Access Protocol) interface for SpaceWire.
 *
 * @standard ECSS-E-ST-50-52C (RMAP)
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef RMAP_H
#define RMAP_H

#include <stdint.h>

/* RMAP command types */
#define RMAP_CMD_READ           0x01U
#define RMAP_CMD_WRITE          0x02U
#define RMAP_CMD_RMW            0x03U   /**< Read-Modify-Write           */
#define RMAP_CMD_WRITE_REPLY    0x04U   /**< Write with reply            */

/* RMAP return codes */
#define RMAP_OK                 0
#define RMAP_ERR_PARAM          (-1)
#define RMAP_ERR_CRC            (-2)
#define RMAP_ERR_TIMEOUT        (-3)
#define RMAP_ERR_COMMAND        (-4)

/**
 * @brief Initialize RMAP target on a SpaceWire port.
 * @param[in] spw_port   SpaceWire port number (0–3).
 * @param[in] dest_key   RMAP destination key for authentication.
 * @return RMAP_OK on success.
 */
int32_t rmap_init(uint8_t spw_port, uint8_t dest_key);

/**
 * @brief Process an incoming RMAP command.
 * @param[in]  packet     Received SpW packet data.
 * @param[in]  pkt_len    Packet length.
 * @param[out] reply      Reply packet buffer.
 * @param[out] reply_len  Reply length.
 * @return RMAP_OK on success.
 */
int32_t rmap_process_command(const uint8_t *packet, uint32_t pkt_len,
                             uint8_t *reply, uint32_t *reply_len);

/**
 * @brief Compute RMAP CRC-8 over a data block.
 * @param[in] data Data bytes.
 * @param[in] len  Number of bytes.
 * @return CRC-8 value.
 */
uint8_t rmap_crc8(const uint8_t *data, uint32_t len);

/**
 * @brief Verify RMAP CRC on a packet.
 * @param[in] packet Packet data.
 * @param[in] len    Packet length.
 * @return RMAP_OK if CRC valid, RMAP_ERR_CRC if invalid.
 */
int32_t rmap_verify_crc(const uint8_t *packet, uint32_t len);

#endif /* RMAP_H */
