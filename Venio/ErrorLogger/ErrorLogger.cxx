#include "ErrorLogger.hxx"

ErrorLogger &ErrorLogger::getInstance()
{
    static ErrorLogger instance;
    return instance;
}

void ErrorLogger::logError(const std::string &message)
{
    std::cerr << "Error: " << message << std::endl;
    system("pause");
    exit(0);
}
