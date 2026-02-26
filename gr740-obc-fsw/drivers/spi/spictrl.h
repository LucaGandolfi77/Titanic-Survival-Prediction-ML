/**
 * @file spictrl.h
 * @brief SPICTRL SPI master driver interface for GR740.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef SPICTRL_H
#define SPICTRL_H

#include <stdint.h>

#define SPI_OK              0
#define SPI_ERR_PARAM       (-1)
#define SPI_ERR_INIT        (-2)
#define SPI_ERR_TIMEOUT     (-3)
#define SPI_ERR_BUSY        (-4)

/**
 * @brief Initialize SPI master controller.
 * @param[in] base_addr SPI base address.
 * @param[in] clock_hz  Desired SPI clock frequency.
 * @return SPI_OK on success.
 */
int32_t spi_init(uint32_t base_addr, uint32_t clock_hz);

/**
 * @brief Select a slave device.
 * @param[in] slave Slave index (0-based).
 * @return SPI_OK on success.
 */
int32_t spi_select(uint8_t slave);

/**
 * @brief Deselect all slave devices.
 * @return SPI_OK on success.
 */
int32_t spi_deselect(void);

/**
 * @brief Transfer one byte (full duplex).
 * @param[in]  tx_byte Byte to transmit.
 * @param[out] rx_byte Pointer to store received byte (may be NULL).
 * @return SPI_OK on success.
 */
int32_t spi_transfer_byte(uint8_t tx_byte, uint8_t *rx_byte);

/**
 * @brief Transfer a buffer (full duplex).
 * @param[in]  tx_buf TX data (NULL = send 0xFF).
 * @param[out] rx_buf RX data (NULL = discard).
 * @param[in]  len    Transfer length.
 * @return SPI_OK on success.
 */
int32_t spi_transfer(const uint8_t *tx_buf, uint8_t *rx_buf, uint32_t len);

/**
 * @brief Write data to SPI (TX only, discard RX).
 * @param[in] data Data to write.
 * @param[in] len  Length.
 * @return SPI_OK on success.
 */
int32_t spi_write(const uint8_t *data, uint32_t len);

/**
 * @brief Read data from SPI (send 0xFF, capture RX).
 * @param[out] data Buffer for read data.
 * @param[in]  len  Length.
 * @return SPI_OK on success.
 */
int32_t spi_read(uint8_t *data, uint32_t len);

#endif /* SPICTRL_H */
