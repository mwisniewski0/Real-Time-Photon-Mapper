#pragma once

// These functions are used to create SDL messages that get sent out to the current
// SDL window every periodMs. The message will have the provided messageCode.
void registerPeriodicalSDLMessage(int periodMs, int messageCode);
void removePeriodicalSDLMessage(int messageCode);

