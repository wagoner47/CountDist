#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/log/expressions/formatters/named_scope.hpp>
#include <boost/log/expressions/formatters/stream.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include "logging.h"

namespace logging = boost::log;
namespace expr = boost::log::expressions;
namespace keywords = boost::log::keywords;
namespace attrs = boost::log::attributes;

using namespace std;

BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", severity_level);

BOOST_LOG_GLOBAL_LOGGER_INIT(logger, logger_t) {
	logger_t lg;

	logging::add_common_attributes();
	logging::core::get()->add_global_attribute("Scope", attrs::named_scope());
	
	logging::add_console_log(std::cout,
			keywords::format = (expr::stream 
				<< expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %T")
				<< " " << severity << " - " 
				<< expr::format_named_scope("Scope", "%n (%F:%l)")
				<< ": " << expr::message
			),
			keywords::filter = (severity >= static_cast<severity_level>(LOG_LEVEL)),
			keywords::auto_flush = true
	);
	
	return lg;
}
