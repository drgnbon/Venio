#pragma once

#include <iostream>
#include <string>

class ErrorLogger
{
public:
    static ErrorLogger &getInstance();
    void logError(const std::string &message);

private:
    ErrorLogger() = default;
    ~ErrorLogger() = default;

    ErrorLogger(const ErrorLogger &) = delete;
    ErrorLogger &operator=(const ErrorLogger &) = delete;
};
