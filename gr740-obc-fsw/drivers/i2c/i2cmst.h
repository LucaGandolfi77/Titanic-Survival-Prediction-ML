/**
 * @file i2cmst.h
 * @brief I2CMST I2C master driver interface for GR740.
 *
 * Supports standard (100 kHz) and fast (400 kHz) modes.
 * Used for EPS power monitors (INA226), temperature sensors,
 * IMU (BMI160), and magnetometer access.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef I2CMST_H
#define I2CMST_H

#include <stdint.h>

#define I2C_OK              0
#define I2C_ERR_PARAM       (-1)
#define I2C_ERR_INIT        (-2)
#define I2C_ERR_NACK        (-3)
#define I2C_ERR_TIMEOUT     (-4)
#define I2C_ERR_ARB_LOST    (-5)
#define I2C_ERR_BUS         (-6)

/**
 * @brief Initialize I2C master controller.
 * @param[in] base_addr I2C base address.
 * @param[in] clock_hz  Desired SCL frequency (100000 or 400000).
 * @return I2C_OK on success.
 */
int32_t i2c_init(uint32_t base_addr, uint32_t clock_hz);

/**
 * @brief Write data to an I2C slave.
 * @param[in] slave_addr 7-bit slave address.
 * @param[in] data       Data to write.
 * @param[in] len        Number of bytes.
 * @return I2C_OK on success.
 */
int32_t i2c_write(uint8_t slave_addr, const uint8_t *data, uint32_t len);

/**
 * @brief Read data from an I2C slave.
 * @param[in]  slave_addr 7-bit slave address.
 * @param[out] data       Buffer for received data.
 * @param[in]  len        Number of bytes to read.
 * @return I2C_OK on success.
 */
int32_t i2c_read(uint8_t slave_addr, uint8_t *data, uint32_t len);

/**
 * @brief Write then read (repeated start) I2C transaction.
 * @param[in]  slave_addr 7-bit slave address.
 * @param[in]  tx_data    Data to write.
 * @param[in]  tx_len     Write length.
 * @param[out] rx_data    Buffer for read data.
 * @param[in]  rx_len     Read length.
 * @return I2C_OK on success.
 */
int32_t i2c_write_read(uint8_t slave_addr,
                        const uint8_t *tx_data, uint32_t tx_len,
                        uint8_t *rx_data, uint32_t rx_len);

/**
 * @brief Read a single register from a device.
 * @param[in]  slave_addr 7-bit slave address.
 * @param[in]  reg_addr   Register address.
 * @param[out] value      Pointer to store register value.
 * @return I2C_OK on success.
 */
int32_t i2c_read_reg(uint8_t slave_addr, uint8_t reg_addr, uint8_t *value);

/**
 * @brief Write a single register to a device.
 * @param[in] slave_addr 7-bit slave address.
 * @param[in] reg_addr   Register address.
 * @param[in] value      Value to write.
 * @return I2C_OK on success.
 */
int32_t i2c_write_reg(uint8_t slave_addr, uint8_t reg_addr, uint8_t value);

#endif /* I2CMST_H */
