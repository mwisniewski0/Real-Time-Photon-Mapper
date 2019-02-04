#include "Helpers.h"
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>
#include <map>

std::string readFileToString(const std::string& path)
{
	std::ifstream file(path);
	std::stringstream ss;
	ss << file.rdbuf();
	file.close();
	return ss.str();
}

Uint32 stateUpdateTimerCallback(Uint32 interval, void *param)
{
	int* code = (int*) param;

	SDL_Event event;
	SDL_UserEvent userevent;

	userevent.type = SDL_USEREVENT;
	userevent.code = *code;

	event.type = SDL_USEREVENT;
	event.user = userevent;

	SDL_PushEvent(&event);

	return interval;
}

struct SdlTimerInfo
{
	int* code;
	SDL_TimerID id;
};

// We are maintaining the status of the timers globally, since SDL does the same.
std::map<int, SdlTimerInfo> registeredSDLMessageTimers;

void registerPeriodicalSDLMessage(int periodMs, int messageCode)
{
	if (mapHasKey(registeredSDLMessageTimers, messageCode))
	{
		throw std::runtime_error("A periodical message with this messageCode has already been "
								 "registered");
	}

	// This will be freed in the callback
	int* code = new int;
	*code = messageCode;
	registeredSDLMessageTimers[messageCode] = {
		code,
		SDL_AddTimer(periodMs, stateUpdateTimerCallback, code)
	};
}

void removePeriodicalSDLMessage(int messageCode)
{
	if (!mapHasKey(registeredSDLMessageTimers, messageCode))
	{
		throw std::runtime_error("A periodical message with this messageCode has already been "
			"registered");
	}
	auto info = registeredSDLMessageTimers[messageCode];
	delete info.code;
	SDL_RemoveTimer(info.id);
	registeredSDLMessageTimers.erase(messageCode);
}
