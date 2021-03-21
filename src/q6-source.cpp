#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <complex>
#include <limits>

using namespace std;
using Complex = complex<double>;

constexpr double eps = 0.0001;  // epsilon for ending steepest descent method
constexpr double eps2 = 1e-8;   // epsilon for comparing against zero

// Functions to safely compare floating and complex numbers against zero
bool isZero(double x) {return abs(x) < eps2;}
bool isZero(Complex z) {return abs(z) < eps2;}

// Class to represent a point or a vector in two dimensions
class Point {
public:
    double x1 = 0;  // first coordinate
    double x2 = 0;  // second coordinate

    Point() = default;
    Point(double x1, double x2): x1(x1), x2(x2) {}

    // Overloading operators for adding two points and multiplying a point with a scalar
    Point operator+(const Point& rhs) const {
        return Point(x1 + rhs.x1, x2 + rhs.x2);
    }
    Point& operator+=(const Point& rhs) {
        x1 += rhs.x1;
        x2 += rhs.x2;
        return *this;
    }
    Point operator*(double sc) const {
        return Point(x1*sc, x2*sc);
    }
    Point& operator*=(double sc) {
        x1 *= sc;
        x2 *= sc;
        return *this;
    }

    // Returns length of the vector
    double length() const {
        return sqrt(pow(x1, 2) + pow(x2, 2));
    }

    // Normalizes the vector by dividing x1 and x2 by length
    void normalize() {
        double len = length();
        x1 /= len;
        x2 /= len;
    }
};

// Class to represent polynomials of any degree.
class Polynomial {
    map<int, double> co;    // associating powers with their coefficients (eg: if polynomial has term 3x^2 then co[2]=3)
public:
    Polynomial() = default;
    Polynomial(const map<int, double>& co): co(co) {
        cleanZeroTerms();
    }
    Polynomial(map<int, double>&& co): co(move(co)) {
        cleanZeroTerms();
    }
    static Polynomial fromConst(double x0) {
        return Polynomial(map<int,double>{{0, x0}});
    }

    // Overloading operators for adding and multiplying two polynomials and raising polynomial to an integer degree
    Polynomial operator+(const Polynomial& rhs) const {
        map<int, double> newCo = this->co;
        for(const auto& p : rhs.co) {
            newCo[p.first] += p.second;
        }
        return Polynomial(move(newCo));
    }
    Polynomial& operator+=(const Polynomial& rhs) {
        for(const auto& p : rhs.co) {
            this->co[p.first] += p.second;
        }
        cleanZeroTerms();
        return *this;
    }
    Polynomial operator*(const Polynomial& rhs) const{
        map<int, double> newCo;
        for(const auto& p1 : co) {
            for(const auto& p2 : rhs.co) {
                newCo[p1.first+p2.first] += p1.second*p2.second;    //c1x^n1 * c2x^n2 = (c1c2)x^(n1+n2)
            }
        }
        return Polynomial(move(newCo));
    }
    Polynomial& operator*=(const Polynomial& rhs) {
        *this = *this * rhs;
        return *this;
    }
    Polynomial operator^(int n) const {
        auto pol = Polynomial::fromConst(1);
        for(int i=0; i<n; i++) {
            pol *= *this;
        }
        return pol;
    }

    // Returns value of polynomial at a point
    double value(double x) const {
        double result = 0;
        for(const auto& p : co) {
            result += p.second * pow(x, p.first);
        }
        return result;
    }

    // Returns derivative of the polynomial
    Polynomial derivative() const {
        map<int, double> newCo;
        for(const auto& p : co) {
            newCo[p.first-1] = p.first*p.second;        // cx^n -> (c*n)x^(n-1)
        }
        newCo.erase(-1);
        return Polynomial(move(newCo));
    }

    // Finds all real roots of the polynomial. Simply applies linear, quadratic, or cubic formula
    // Throws when there are no real roots or degree of polynomial is > 3 (4 is too difficult, 5+ has no formula)
    vector<double> findRealRoots() {
        int degree = co.empty() ? 0 : co.rbegin()->first;
        if(degree == 0) {
            throw runtime_error("No roots");
        } else if(degree == 1) {
            if(isZero(co[1])) throw runtime_error("No roots");
            return {-co[0] / co[1]};
        } else if(degree == 2) {
            double a(co[2]), b(co[1]), c(co[0]);
            double delta = b*b - 4*a*c;
            if(delta < 0) throw runtime_error("No roots");
            return {-b-sqrt(delta)/2*a, -b+sqrt(delta)/2*a};
        } else if(degree == 3) {
            double a(co[3]), b(co[2]), c(co[1]), d(co[0]);
            Complex xi = -0.5 + 0.866025403784439i;
            Complex delta0 = b*b - 3*a*c;
            Complex delta1 = 2*b*b*b - 9*a*b*c + 27*a*a*d;
            Complex C = pow((delta1 + pow(pow(delta1, 2) - pow(delta0,3) * 4., 1./2)) / 2., 1./3);
            if(isZero(C)) C = pow((delta1 - pow(pow(delta1, 2) - pow(delta0,3) * 4., 1./2)) / 2., 1./3);
            vector<double> roots;
            for(int i=0; i<3; i++) {
                Complex x = -1./(3*a) * (b + pow(xi,i)*C + delta0/(pow(xi,i)*C));
                if(isZero(x.imag())) roots.push_back(x.real());
            }
            return roots;
        } else {
            throw runtime_error("Too difficult");
        }
    }

    // Minimizes the polynomial by finding all real roots of p' and picking the one with the smallest value
    // Returns x which minimizes the polynomial, not the value
    double minimize() {
        auto roots = derivative().findRealRoots();
        double minX = 0;
        double minVal = numeric_limits<double>::max();
        for(double root : roots) {
            double val = this->value(root);
            if(val < minVal) {
                minX = root;
                minVal = val;
            }
        }
        return minX;
    }

    // Deletes all 0x^n terms from coefficients map
    void cleanZeroTerms() {
        auto itr = co.begin();
        while(itr != co.end()) {
            if(isZero(itr->second)) itr = co.erase(itr);
            else itr++;
        }
    }
};

// Class to represent a function of two variables which includes only terms c*x1^a*x2^b
class BinaryF {
public:
    vector<vector<double>> c;   // c[a][b] is the coefficient of x1^a*x2^b
    int x1deg = 0;              // highest degree of x1 in all terms
    int x2deg = 0;              // highest degree of x2 in all terms

    BinaryF() = default;
    BinaryF(int x1deg, int x2deg): c(x1deg+1, vector<double>(x2deg+1, 0)), x1deg(x1deg), x2deg(x2deg) {}
    BinaryF(const vector<vector<double>>& c): c(c), x1deg(c.size()-1), x2deg(c.at(0).size()-1) {}
    
    // Returns value of function at a point (p.x1, p.x2)
    double value(const Point& p) const {
        double result = 0;
        for(int i=0; i<=x1deg; i++) {
            for(int j=0; j<=x2deg; j++) {
                result += c[i][j] * pow(p.x1, i) * pow(p.x2,j);
            }
        }
        return result;
    }

    // Returns value of function at a point where coordinates of points are polynomials
    // Result is a polynomial too
    Polynomial value(const Polynomial& p1, const Polynomial& p2) const {
        Polynomial result;
        for(int i=0; i<=x1deg; i++) {
            for(int j=0; j<=x2deg; j++) {
                result += Polynomial::fromConst(c[i][j]) * (p1^i) * (p2^j);
            }
        }
        return result;
    }

    // Returns gradient of the function, which is a pair of functions
    pair<BinaryF, BinaryF> gradient() const {           
        BinaryF f1(x1deg, x2deg);   // partial f / partial x1
        BinaryF f2(x1deg, x2deg);   // partial f / partial x2
        for(int i=0; i<=x1deg; i++) {
            for(int j=0; j<=x2deg; j++) {
                if(i != 0) f1.c[i-1][j] += c[i][j] * i;     // c*x1^i*x2^j -> c*i*x1^(i-1)*x2^j
                if(j != 0) f2.c[i][j-1] += c[i][j] * j;     // c*x1^i*x2^j -> c*j*x1^i*x2^(j-1)
            }
        }
        return {move(f1), move(f2)};
    }
};

// Applies steepest descent method.
// Finds gradient. Then at each iteration calculates direction (from gradient) and alpha (by minimizing f(p+alpha*d))
// Updates p
// Stops when length of gradient of p is less than epsilon
void steepestDescentMethod(const BinaryF& func, Point& p) {
    pair<BinaryF, BinaryF> grad = func.gradient();
    int i=0;
    while(1) {
        i++;
        printf("Iteration: %d, Point: (%f, %f), Value: %f\n", i, p.x1, p.x2, func.value(p));
        Point gradAtPoint = {grad.first.value(p), grad.second.value(p)};
        if(gradAtPoint.length() < eps) break;
        Point direction = gradAtPoint * (-1);
        direction.normalize();
        Polynomial minimizeThis = func.value(
                Polynomial({{0, p.x1}, {1, direction.x1}}),
                Polynomial({{0, p.x2}, {1, direction.x2}})
        );
        double alpha = minimizeThis.minimize();
        p += direction * alpha;
    }
}

int main() {
    BinaryF func({
        {16,    -2,     0,      0,      1},
        {-3,    0,      0,      -20,    0},
        {1,     0,      150,    0,      0},
        {0,     -500,   0,      0,      0},
        {625,   0,      0,      0,      0}
    });
    Point p(1, 2);
    printf("Initial point: (%f, %f), Initial Value: %f\n", p.x1, p.x2, func.value(p));
    steepestDescentMethod(func, p);
    printf("Steepest descent method completed.\n");
    printf("Minimum value of f is %f, achieved at (%f, %f)\n", func.value(p), p.x1, p.x2);
}