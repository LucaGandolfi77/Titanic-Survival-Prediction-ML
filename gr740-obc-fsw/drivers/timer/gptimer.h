/**
 * @file gptimer.h
 * @brief GPTIMER general-purpose timer driver interface for GR740.
 *
 * Provides multi-channel hardware timer support. Timer 0 is reserved
 * for BSP systick (1 ms). Remaining timers available for:
 *   - Watchdog feeding
 *   - Timeout guards
 *   - Periodic scheduling triggers
 *   - Timestamping
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef GPTIMER_H
#define GPTIMER_H

#include <stdint.h>

#define TIMER_OK            0
#define TIMER_ERR_PARAM     (-1)
#define TIMER_ERR_INIT      (-2)
#define TIMER_ERR_BUSY      (-3)

/** Maximum number of timer units */
#define TIMER_MAX_UNITS     2U
/** Maximum timers per unit */
#define TIMER_MAX_CHANNELS  4U

/** Timer callback type */
typedef void (*timer_callback_t)(uint32_t unit, uint32_t channel);

/**
 * @brief Initialize a timer unit.
 * @param[in] unit      Timer unit index (0 or 1).
 * @param[in] base_addr Timer base address.
 * @return TIMER_OK on success.
 */
int32_t timer_init(uint32_t unit, uint32_t base_addr);

/**
 * @brief Configure a timer channel for periodic interrupt.
 * @param[in] unit      Timer unit.
 * @param[in] channel   Channel (0-3). Channel 0 of unit 0 reserved for systick.
 * @param[in] period_us Period in microseconds.
 * @param[in] cb        Callback on expiry (NULL for no callback).
 * @return TIMER_OK on success.
 */
int32_t timer_configure(uint32_t unit, uint32_t channel,
                         uint32_t period_us, timer_callback_t cb);

/**
 * @brief Start a configured timer channel.
 * @param[in] unit    Timer unit.
 * @param[in] channel Channel.
 * @return TIMER_OK on success.
 */
int32_t timer_start(uint32_t unit, uint32_t channel);

/**
 * @brief Stop a timer channel.
 * @param[in] unit    Timer unit.
 * @param[in] channel Channel.
 * @return TIMER_OK on success.
 */
int32_t timer_stop(uint32_t unit, uint32_t channel);

/**
 * @brief Read current counter value.
 * @param[in]  unit    Timer unit.
 * @param[in]  channel Channel.
 * @param[out] value   Pointer to store counter value.
 * @return TIMER_OK on success.
 */
int32_t timer_read_counter(uint32_t unit, uint32_t channel, uint32_t *value);

/**
 * @brief Configure a timer channel as a one-shot timeout.
 * @param[in] unit       Timer unit.
 * @param[in] channel    Channel.
 * @param[in] timeout_us Timeout in microseconds.
 * @param[in] cb         Callback on expiry.
 * @return TIMER_OK on success.
 */
int32_t timer_oneshot(uint32_t unit, uint32_t channel,
                       uint32_t timeout_us, timer_callback_t cb);

/**
 * @brief Timer ISR — call from interrupt handler for appropriate timer.
 * @param[in] unit Timer unit that generated interrupt.
 */
void timer_isr(uint32_t unit);

/**
 * @brief Get free-running timestamp (unit 1, channel 3 counter).
 * @return Counter value in timer ticks.
 */
uint32_t timer_get_timestamp(void);

#endif /* GPTIMER_H */
