#include<iostream>
#include<conio.h>
using namespace std;


class shape
{
	char *name;

public:
	shape(char* name)
	{
		this->name = name;
	}

	float get_area();

	char* get_name()
	{
		return this->name;
	}
};

//a circle class extending shape class
class circle : public shape
{
	int radius;

public:
	circle(int radius) : shape("circle")
	{
		this->radius = radius;
	}

	float get_area()
	{
		return 3.14 * radius * radius;
	}
};

//a square class extending shape class
class square : public shape
{
	int a;

public:
	square(int a) : shape("square")
	{
		this->a = a;
	}

	float get_area()
	{
		return a*a;
	}
	
};

//a triangle class extending shape class
class triangle : public shape
{
	int b, h;

public:
	triangle(int b, int h) : shape("triangle")
	{
		this->b = b;
		this->h = h;
	}

	float get_area()
	{
		return 0.5 * b * h;
	}
};


int main()
{
	triangle* t = new triangle(2, 4);
	cout << t->get_name() << " " << t->get_area();
	return 0;
}