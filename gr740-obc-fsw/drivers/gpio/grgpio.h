/**
 * @file grgpio.h
 * @brief GRGPIO general-purpose I/O driver interface for GR740.
 *
 * Used for discrete I/O, deployment mechanisms, status LEDs,
 * and interrupt-capable input lines.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef GRGPIO_H
#define GRGPIO_H

#include <stdint.h>

#define GPIO_OK             0
#define GPIO_ERR_PARAM      (-1)
#define GPIO_ERR_INIT       (-2)

#define GPIO_DIR_INPUT      0U
#define GPIO_DIR_OUTPUT     1U

#define GPIO_LOW            0U
#define GPIO_HIGH           1U

#define GPIO_IRQ_NONE       0U
#define GPIO_IRQ_RISING     1U
#define GPIO_IRQ_FALLING    2U
#define GPIO_IRQ_BOTH       3U
#define GPIO_IRQ_LEVEL      4U

/** Maximum number of GPIO pins */
#define GPIO_MAX_PINS       32U

/** GPIO interrupt callback type */
typedef void (*gpio_callback_t)(uint32_t pin);

/**
 * @brief Initialize GPIO controller.
 * @param[in] base_addr GPIO base address.
 * @return GPIO_OK on success.
 */
int32_t gpio_init(uint32_t base_addr);

/**
 * @brief Set pin direction.
 * @param[in] pin  Pin number (0-31).
 * @param[in] dir  GPIO_DIR_INPUT or GPIO_DIR_OUTPUT.
 * @return GPIO_OK on success.
 */
int32_t gpio_set_direction(uint32_t pin, uint32_t dir);

/**
 * @brief Write a pin output value.
 * @param[in] pin   Pin number (0-31).
 * @param[in] value GPIO_LOW or GPIO_HIGH.
 * @return GPIO_OK on success.
 */
int32_t gpio_write(uint32_t pin, uint32_t value);

/**
 * @brief Read a pin value.
 * @param[in]  pin   Pin number (0-31).
 * @param[out] value Pointer to store pin state.
 * @return GPIO_OK on success.
 */
int32_t gpio_read(uint32_t pin, uint32_t *value);

/**
 * @brief Toggle a pin output.
 * @param[in] pin Pin number (0-31).
 * @return GPIO_OK on success.
 */
int32_t gpio_toggle(uint32_t pin);

/**
 * @brief Write all outputs at once (32-bit mask).
 * @param[in] value 32-bit output value.
 */
void gpio_write_all(uint32_t value);

/**
 * @brief Read all inputs at once.
 * @return 32-bit input value.
 */
uint32_t gpio_read_all(void);

/**
 * @brief Configure pin interrupt.
 * @param[in] pin  Pin number.
 * @param[in] mode Interrupt mode (GPIO_IRQ_*).
 * @param[in] cb   Callback function (NULL to disable).
 * @return GPIO_OK on success.
 */
int32_t gpio_irq_configure(uint32_t pin, uint32_t mode, gpio_callback_t cb);

/**
 * @brief GPIO ISR — call from interrupt handler.
 */
void gpio_isr(void);

#endif /* GRGPIO_H */
