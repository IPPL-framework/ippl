#ifndef FIELDDEBUGFUNCTIONS_H
#define FIELDDEBUGFUNCTIONS_H

class Inform;

void setInform(Inform& inform);
void setFormat(int ElementsPerLine, int DigitsPastDecimal, 
	       int WidthOfElements = 0);

#endif
