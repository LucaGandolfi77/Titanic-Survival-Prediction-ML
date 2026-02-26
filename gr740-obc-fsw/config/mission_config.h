/**
 * @file mission_config.h
 * @brief Mission-level configuration parameters for GR740 OBC FSW.
 *
 * This file defines all mission-configurable parameters including
 * software version, timing, APID assignments, and operational limits.
 *
 * @standard ECSS-E-ST-40C, ECSS-E-ST-70-41C (PUS-C)
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef MISSION_CONFIG_H
#define MISSION_CONFIG_H

#include <stdint.h>

/* ======================================================================
 * SOFTWARE VERSION
 * ====================================================================== */
#define FSW_VERSION_MAJOR       1U
#define FSW_VERSION_MINOR       0U
#define FSW_VERSION_PATCH       0U
#define FSW_VERSION_STRING      "1.0.0"

/* ======================================================================
 * SPACECRAFT IDENTIFICATION
 * ====================================================================== */
#define SPACECRAFT_ID           0x01U       /**< Spacecraft identifier */
#define MISSION_NAME            "GR740-SAT" /**< Mission name string   */

/* ======================================================================
 * CCSDS APID ASSIGNMENTS (11-bit, 0x000–0x7FF)
 * ====================================================================== */
#define APID_OBC_HK            0x010U  /**< OBC Housekeeping telemetry   */
#define APID_OBC_EVENT         0x011U  /**< OBC Event reports            */
#define APID_OBC_TIME          0x012U  /**< OBC Time reports             */
#define APID_OBC_CMD           0x020U  /**< OBC Telecommand APID         */
#define APID_ADCS_HK           0x030U  /**< ADCS Housekeeping            */
#define APID_ADCS_CMD          0x031U  /**< ADCS Telecommands            */
#define APID_EPS_HK            0x040U  /**< EPS Housekeeping             */
#define APID_EPS_CMD           0x041U  /**< EPS Telecommands             */
#define APID_PAYLOAD_HK        0x050U  /**< Payload Housekeeping         */
#define APID_PAYLOAD_DATA      0x051U  /**< Payload Science Data         */
#define APID_PAYLOAD_CMD       0x052U  /**< Payload Telecommands         */
#define APID_IDLE              0x7FFU  /**< Idle packet (all ones)        */

/* ======================================================================
 * TIMING PARAMETERS
 * ====================================================================== */
#define MINOR_FRAME_MS          100U    /**< Minor frame period (ms)      */
#define MAJOR_FRAME_MS          1000U   /**< Major frame period (ms)      */
#define MINOR_FRAMES_PER_MAJOR  10U     /**< Minor frames per major frame */
#define SYSTEM_TICK_HZ          1000U   /**< System tick rate (Hz)        */

/* ======================================================================
 * HOUSEKEEPING PARAMETERS
 * ====================================================================== */
#define HK_REPORT_INTERVAL_S    5U      /**< Default HK report interval   */
#define HK_MAX_PARAMETERS       64U     /**< Max parameters in HK report  */

/* ======================================================================
 * WATCHDOG PARAMETERS
 * ====================================================================== */
#define WDG_HW_TIMEOUT_MS       2000U   /**< Hardware watchdog timeout    */
#define WDG_SW_CHECK_MS         500U    /**< Software watchdog check rate */
#define WDG_MISS_THRESHOLD      3U      /**< Consecutive misses for FDIR  */
#define WDG_MAX_TASKS           16U     /**< Max tasks to monitor         */

/* ======================================================================
 * FDIR PARAMETERS
 * ====================================================================== */
#define FDIR_MAX_FAULTS         32U     /**< Max fault definitions        */
#define FDIR_MAX_ERROR_LOG      128U    /**< Max error log entries        */
#define FDIR_TEMP_HIGH_LIMIT    85      /**< OBC temp high limit (°C)     */
#define FDIR_TEMP_LOW_LIMIT     (-40)   /**< OBC temp low limit (°C)      */
#define FDIR_VOLTAGE_HIGH_MV    3600U   /**< Voltage high limit (mV)      */
#define FDIR_VOLTAGE_LOW_MV     3000U   /**< Voltage low limit (mV)       */

/* ======================================================================
 * COMMUNICATION PARAMETERS
 * ====================================================================== */
#define CAN_NODE_ID             0x01U   /**< OBC CAN node ID              */
#define CAN_BAUDRATE            1000000U /**< CAN bus baudrate (1 Mbps)   */
#define SPW_LINK_SPEED_MBPS     100U    /**< SpaceWire link speed         */
#define UART_BAUDRATE           115200U /**< UART baud rate               */
#define SPI_CLOCK_HZ            10000000U /**< SPI clock (10 MHz)         */
#define I2C_CLOCK_HZ            400000U /**< I2C clock (400 kHz)          */

/* ======================================================================
 * BUFFER SIZES (power of 2)
 * ====================================================================== */
#define CAN_TX_RING_SIZE        64U     /**< CAN TX ring buffer entries   */
#define CAN_RX_RING_SIZE        64U     /**< CAN RX ring buffer entries   */
#define SPW_TX_DESC_COUNT       16U     /**< SpW TX DMA descriptors       */
#define SPW_RX_DESC_COUNT       16U     /**< SpW RX DMA descriptors       */
#define SPW_MAX_PACKET_SIZE     4096U   /**< SpW max packet size bytes    */
#define UART_TX_BUF_SIZE        256U    /**< UART TX buffer bytes         */
#define UART_RX_BUF_SIZE        512U    /**< UART RX buffer bytes         */
#define CCSDS_MAX_PACKET_SIZE   1024U   /**< Max CCSDS space packet       */
#define TM_FRAME_SIZE           1115U   /**< TM transfer frame size       */
#define TC_FRAME_SIZE           1024U   /**< TC transfer frame size       */

/* ======================================================================
 * VIRTUAL CHANNEL ASSIGNMENT (TM)
 * ====================================================================== */
#define VC_REALTIME_HK          0U      /**< Real-time housekeeping       */
#define VC_STORED_HK            1U      /**< Stored housekeeping          */
#define VC_EVENT                2U      /**< Event data                   */
#define VC_SCIENCE              3U      /**< Science data                 */
#define VC_IDLE                 7U      /**< Idle virtual channel         */
#define VC_COUNT                8U      /**< Total virtual channels       */

/* ======================================================================
 * NVM PARAMETERS
 * ====================================================================== */
#define NVM_PARAM_TABLE_SIZE    2048U   /**< Parameter table size (bytes) */
#define NVM_MAX_PARAMS          128U    /**< Max storable parameters      */
#define NVM_MAGIC               0xA5C3E7F1U /**< NVM validity magic      */

/* ======================================================================
 * TASK PARAMETERS
 * ====================================================================== */
#define TASK_STACK_SIZE         4096U   /**< Default task stack (bytes)   */
#define TASK_MAX_COUNT          16U     /**< Maximum number of tasks      */

/* Priority levels (1=highest, 255=lowest for RTEMS) */
#define TASK_PRIO_WATCHDOG      2U      /**< Watchdog task priority       */
#define TASK_PRIO_SCHEDULER     3U      /**< Scheduler task priority      */
#define TASK_PRIO_TC_HANDLER    5U      /**< TC handler task priority     */
#define TASK_PRIO_TM_GEN        6U      /**< TM generation priority       */
#define TASK_PRIO_HK            8U      /**< Housekeeping priority        */
#define TASK_PRIO_FDIR          4U      /**< FDIR task priority           */
#define TASK_PRIO_THERMAL       10U     /**< Thermal control priority     */
#define TASK_PRIO_BACKGROUND    20U     /**< Background task priority     */

/* ======================================================================
 * TIME-TAGGED COMMAND QUEUE
 * ====================================================================== */
#define TTC_MAX_COMMANDS        32U     /**< Max time-tagged commands     */

#endif /* MISSION_CONFIG_H */
