/**
 * @file pus_st01.h
 * @brief PUS Service 1 — Request Verification (ECSS-E-ST-70-41C).
 *
 * Generates TM(1,1) acceptance success, TM(1,2) acceptance failure,
 * TM(1,7) execution success, TM(1,8) execution failure reports.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef PUS_ST01_H
#define PUS_ST01_H

#include <stdint.h>
#include "../ccsds/space_packet.h"

/** PUS Service 1 subtypes */
#define PUS_ST01_ACCEPT_OK      1U  /**< TM(1,1) Acceptance success  */
#define PUS_ST01_ACCEPT_FAIL    2U  /**< TM(1,2) Acceptance failure  */
#define PUS_ST01_START_OK       3U  /**< TM(1,3) Start success       */
#define PUS_ST01_START_FAIL     4U  /**< TM(1,4) Start failure       */
#define PUS_ST01_PROGRESS       5U  /**< TM(1,5) Progress            */
#define PUS_ST01_PROGRESS_FAIL  6U  /**< TM(1,6) Progress failure    */
#define PUS_ST01_EXEC_OK        7U  /**< TM(1,7) Execution success   */
#define PUS_ST01_EXEC_FAIL      8U  /**< TM(1,8) Execution failure   */

/** Error codes for verification */
#define PUS_ERR_NONE            0x0000U
#define PUS_ERR_ILLEGAL_APID    0x0001U
#define PUS_ERR_ILLEGAL_TYPE    0x0002U
#define PUS_ERR_ILLEGAL_SUBTYPE 0x0003U
#define PUS_ERR_ILLEGAL_LEN     0x0004U
#define PUS_ERR_CRC_FAIL        0x0005U
#define PUS_ERR_EXEC_TIMEOUT    0x0010U
#define PUS_ERR_EXEC_FAIL       0x0011U
#define PUS_ERR_PARAM_OOR       0x0020U

#define PUS_ST01_OK             0
#define PUS_ST01_ERR_PARAM      (-1)

/**
 * @brief Initialize PUS Service 1.
 * @param[in] apid TM source APID for verification reports.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_init(uint16_t apid);

/**
 * @brief Send acceptance success TM(1,1).
 * @param[in] tc_pkt Pointer to the received TC packet.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_accept_ok(const ccsds_packet_t *tc_pkt);

/**
 * @brief Send acceptance failure TM(1,2).
 * @param[in] tc_pkt   Pointer to the received TC packet.
 * @param[in] err_code Error code.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_accept_fail(const ccsds_packet_t *tc_pkt, uint16_t err_code);

/**
 * @brief Send execution start success TM(1,3).
 * @param[in] tc_pkt Pointer to the received TC packet.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_start_ok(const ccsds_packet_t *tc_pkt);

/**
 * @brief Send execution completion success TM(1,7).
 * @param[in] tc_pkt Pointer to the received TC packet.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_exec_ok(const ccsds_packet_t *tc_pkt);

/**
 * @brief Send execution completion failure TM(1,8).
 * @param[in] tc_pkt   Pointer to the received TC packet.
 * @param[in] err_code Error code.
 * @return PUS_ST01_OK on success.
 */
int32_t pus_st01_exec_fail(const ccsds_packet_t *tc_pkt, uint16_t err_code);

#endif /* PUS_ST01_H */
