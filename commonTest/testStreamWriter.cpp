#include "../external/catch.hpp"
#include <sstream>
#include "../common/streamWriter.h"
#include <iostream>


TEST_CASE("Strings are saved and loaded") {
	std::stringstream s;
	writeToStream<std::string>(s, "test text");
	REQUIRE(readFromStream<std::string>(s) == "test text");
}