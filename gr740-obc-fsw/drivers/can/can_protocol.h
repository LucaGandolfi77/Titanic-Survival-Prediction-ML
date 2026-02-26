/**
 * @file can_protocol.h
 * @brief CAN protocol framing definitions (CANopen-style / CCSDS-over-CAN).
 *
 * Defines CAN frame structures, COB-ID assignments and protocol
 * constants for the OBC CAN bus network.
 *
 * @standard ISO 11898, CANopen (CiA 301)
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef CAN_PROTOCOL_H
#define CAN_PROTOCOL_H

#include <stdint.h>

/* ======================================================================
 * CAN 2.0B Constants
 * ====================================================================== */
#define CAN_MAX_DATA_LEN        8U          /**< Max CAN data bytes      */
#define CAN_STD_ID_MASK         0x7FFU      /**< 11-bit standard ID mask */
#define CAN_EXT_ID_MASK         0x1FFFFFFFU /**< 29-bit extended ID mask */

/* ======================================================================
 * CANopen Function Codes (4-bit, upper bits of 11-bit COB-ID)
 * ====================================================================== */
#define CANOPEN_NMT             0x000U  /**< NMT service                 */
#define CANOPEN_SYNC            0x080U  /**< SYNC message                */
#define CANOPEN_EMCY            0x080U  /**< Emergency (+ node ID)       */
#define CANOPEN_TPDO1           0x180U  /**< TPDO1 (+ node ID)           */
#define CANOPEN_RPDO1           0x200U  /**< RPDO1 (+ node ID)           */
#define CANOPEN_TPDO2           0x280U  /**< TPDO2 (+ node ID)           */
#define CANOPEN_RPDO2           0x300U  /**< RPDO2 (+ node ID)           */
#define CANOPEN_TPDO3           0x380U  /**< TPDO3 (+ node ID)           */
#define CANOPEN_RPDO3           0x400U  /**< RPDO3 (+ node ID)           */
#define CANOPEN_TPDO4           0x480U  /**< TPDO4 (+ node ID)           */
#define CANOPEN_RPDO4           0x500U  /**< RPDO4 (+ node ID)           */
#define CANOPEN_TSDO            0x580U  /**< SDO TX (+ node ID)          */
#define CANOPEN_RSDO            0x600U  /**< SDO RX (+ node ID)          */
#define CANOPEN_HEARTBEAT       0x700U  /**< Heartbeat (+ node ID)       */

/* ======================================================================
 * Satellite Bus Node IDs
 * ====================================================================== */
#define CAN_NODE_OBC            0x01U   /**< OBC node ID                 */
#define CAN_NODE_EPS            0x02U   /**< EPS node ID                 */
#define CAN_NODE_ADCS           0x03U   /**< ADCS node ID                */
#define CAN_NODE_PAYLOAD        0x04U   /**< Payload node ID             */
#define CAN_NODE_PROPULSION     0x05U   /**< Propulsion node ID          */
#define CAN_NODE_THERMAL        0x06U   /**< Thermal node ID             */
#define CAN_NODE_BROADCAST      0x00U   /**< Broadcast (all nodes)       */

/* ======================================================================
 * CAN Frame Structure
 * ====================================================================== */

/**
 * @brief CAN 2.0B frame structure.
 */
typedef struct {
    uint32_t id;            /**< CAN identifier (11-bit or 29-bit)       */
    uint8_t  data[CAN_MAX_DATA_LEN]; /**< Data payload (0-8 bytes)      */
    uint8_t  dlc;           /**< Data Length Code (0-8)                   */
    uint8_t  extended;      /**< 1 = extended (29-bit), 0 = standard     */
    uint8_t  rtr;           /**< 1 = Remote Transmission Request         */
    uint8_t  reserved;      /**< Padding for alignment                   */
} can_frame_t;

/* ======================================================================
 * CAN Protocol Message Types
 * ====================================================================== */

/** NMT state commands */
#define NMT_CMD_START           0x01U   /**< Start remote node           */
#define NMT_CMD_STOP            0x02U   /**< Stop remote node            */
#define NMT_CMD_PREOP           0x80U   /**< Enter pre-operational       */
#define NMT_CMD_RESET_NODE      0x81U   /**< Reset node                  */
#define NMT_CMD_RESET_COMM      0x82U   /**< Reset communication         */

/** Heartbeat states */
#define HB_STATE_BOOTUP         0x00U   /**< Boot-up state               */
#define HB_STATE_STOPPED        0x04U   /**< Stopped state               */
#define HB_STATE_OPERATIONAL    0x05U   /**< Operational state           */
#define HB_STATE_PREOP          0x7FU   /**< Pre-operational state       */

/* ======================================================================
 * Utility Macros
 * ====================================================================== */

/**
 * @brief Build a CANopen COB-ID from function code and node ID.
 * @param func_code CANopen function code (e.g., CANOPEN_TPDO1).
 * @param node_id   Node ID (1-127).
 * @return 11-bit COB-ID.
 */
#define CAN_COB_ID(func_code, node_id) \
    (((uint32_t)(func_code)) | ((uint32_t)(node_id) & 0x7FU))

/**
 * @brief Extract node ID from a COB-ID.
 * @param cob_id 11-bit COB-ID.
 * @return Node ID (0-127).
 */
#define CAN_NODE_FROM_COB(cob_id) \
    ((uint8_t)((cob_id) & 0x7FU))

/**
 * @brief Extract function code from a COB-ID.
 * @param cob_id 11-bit COB-ID.
 * @return Function code.
 */
#define CAN_FUNC_FROM_COB(cob_id) \
    ((uint16_t)((cob_id) & 0x780U))

#endif /* CAN_PROTOCOL_H */
