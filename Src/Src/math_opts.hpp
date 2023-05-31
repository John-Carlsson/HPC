#ifndef MATH_OPTS_H
#define MATH_OPTS_H


class MMAOperation
{
private:
    
protected:

public:
    virtual const char* GetOPTMame() = 0;

    virtual double Import() =  0;

    virtual double Compute() = 0;

    virtual double Export() = 0;
    
    virtual void ComputeNTime(unsigned int loopCount) = 0;

    virtual void Cleanup() = 0;
};

#endif