class ErrorLogger
{
public:
    static ErrorLogger &getInstance()
    {
        static ErrorLogger instance;
        return instance;
    }

    void logError(const std::string &message)
    {
        std::cerr << "Error: " << message << std::endl;
        system("pause");
        exit(0);
    }

private:
    ErrorLogger() = default;
    ~ErrorLogger() = default;

    ErrorLogger(const ErrorLogger &) = delete;
    ErrorLogger &operator=(const ErrorLogger &) = delete;
};