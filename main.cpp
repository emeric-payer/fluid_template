// STATUS: END OF LAB 8 (MAKEUP SESSION)

#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>

#include <vector>
#include <cstdlib>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"

double sqr(double x) { return x * x; };

// Implemented in lab 3
double target_fluid_area = 0.4;
const double PI = 3.141592653589793;

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}


class Polygon {
public:

    double area() {
        if (vertices.size() < 3) return 0;
        // TODO Lab 2
        // Compute the area of the polygon
        
        double total_sum = 0.0;
        int num_v = vertices.size();
        
        for (int i = 0; i < num_v; i++) {
            int next_idx = (i+1) % num_v;
            total_sum += (vertices[i][0]*vertices[next_idx][1] - vertices[next_idx][0]*vertices[i][1]);
        }
        
        if (total_sum < 0) total_sum = -total_sum;
        return 0.5 * total_sum;
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // TODO Lab 3
        // Compute the centroid of the polygon

        int num_v = vertices.size();
        double area_acc = 0.0;
        Vector c_accum(0, 0);

        for (int k = 1; k < num_v - 1; k++) {
            Vector c1 = vertices[0];
            Vector c2 = vertices[k];
            Vector c3 = vertices[k+1];

            double tri_area = 0.5 * ((c2[0]-c1[0])*(c3[1]-c1[1]) - (c2[1]-c1[1])*(c3[0]-c1[0]));

            Vector tri_mid = (c1+c2+c3) / 3.0;
            c_accum = c_accum + tri_area*tri_mid;
            area_acc += tri_area;
        }

        if (area_acc < 1e-12 && area_acc > -1e-12) {
            return vertices[0];
        }

        return c_accum / area_acc;
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;

        // TODO Lab 2
        // Compute the integral of ||x-Pi||^2 over the polygon
    
        double result_integral = 0.0;
        int num_v = vertices.size();
        
        for (int k = 1; k < num_v-1; k++) {
            Vector c1 = vertices[0];
            Vector c2 = vertices[k];
            Vector c3 = vertices[k+1];
            
            Vector d1 = c1 - Pi;
            Vector d2 = c2 - Pi;
            Vector d3 = c3 - Pi;
            
            double local_sum = dot(d1, d1) + dot(d2, d2) + dot(d3, d3) + dot(d1, d2) + dot(d2, d3) + dot(d3, d1);
                               
            double tri_area = 0.5 * ((c2[0]-c1[0]) * (c3[1]-c1[1]) - (c2[1]-c1[1]) * (c3[0]-c1[0]));
            if (tri_area < 0) {
                tri_area = -tri_area;
            }
            
            result_integral += (tri_area/6.0) * local_sum;
        }

        return result_integral;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");

    }

    fprintf(f, "</svg>\n");
    fclose(f);
}


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };


    void compute() {

        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi

        cells.clear();

        for (int i = 0; i < points.size(); i++) {
            Polygon cell;

            cell.vertices.push_back(Vector(0, 0));
            cell.vertices.push_back(Vector(1, 0));
            cell.vertices.push_back(Vector(1, 1));
            cell.vertices.push_back(Vector(0, 1));

            for (int j = 0; j < points.size(); j++) {
                if (i == j) {
                    continue;
                }

                // Line modified during lab 2

                /*
                cell = clip_by_bisector(cell, points[i], points[j], 0, 0);
                */

                cell = clip_by_bisector(cell, points[i], points[j], weights[i], weights[j]);
            }

            // Implemented during lab 3

            if (weights.size() > points.size()) {
                double w_air = weights[points.size()];
                double radius2 = weights[i] - w_air;

                if (radius2 <= 0) {
                    cell.vertices.clear();
                } else {
                    double radius = sqrt(radius2);
                    int disk_sides = 60;

                    for (int s = 0; s < disk_sides; s++) {
                        double a1 = 2.0 * PI * s / disk_sides;
                        double a2 = 2.0 * PI * (s + 1) / disk_sides;

                        Vector u(points[i][0] + radius*cos(a1), points[i][1] + radius*sin(a1));
                        Vector v(points[i][0] + radius*cos(a2), points[i][1] + radius*sin(a2));
                        cell = clip_by_edge(cell, u, v);

                        if (cell.vertices.empty()) {
                            break;
                        }
                    }
                }
            }

            cells.push_back(cell);
        }
    }


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Polygon result;

        int num_v = V.vertices.size();

        if (num_v == 0) {
            return result;
        }

        Vector edge = v - u;

        for (int i = 0; i < num_v; i++) {
            Vector A = V.vertices[i];
            Vector B = V.vertices[(i+1) % num_v];

            double side_a = edge[0]*(A[1]-u[1]) - edge[1]*(A[0]-u[0]);
            double side_b = edge[0]*(B[1]-u[1]) - edge[1]*(B[0]-u[0]);

            bool A_inside = (side_a >= 0);
            bool B_inside = (side_b >= 0);

            double t = 0.0;

            if (side_a != side_b) {
                t = side_a / (side_a-side_b);
            }

            Vector P = A + t * (B - A);

            if (B_inside) {
                if (!A_inside) {
                    result.vertices.push_back(P);
                }
                result.vertices.push_back(B);
            } else if (A_inside) {
                result.vertices.push_back(P);
            }
        }

        return result;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {

        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Polygon result;

        int num_v = V.vertices.size();
        if (num_v == 0) {
            return result;
        }

        Vector normal = Pi - P0;
        double dist2 = normal.norm2();
        
        Vector M = 0.5 * (P0+Pi) + ((w0-wi) / (2.0*dist2)) * (Pi-P0);


        for (int i = 0; i < num_v; i++) {
            Vector A = V.vertices[i];
            Vector B = V.vertices[(i+1) % num_v];

            bool A_inside = ((A-P0).norm2() - w0 <= (A-Pi).norm2() - wi);
            bool B_inside = ((B-P0).norm2() - w0 <= (B-Pi).norm2() - wi);

            double denom = dot(B-A, normal);
            double t = 0.0;
            if (abs(denom) > 1e-11) {
                t = dot(M-A, normal) / denom;
            }
            
            Vector P = A + t * (B-A);
            
            if (B_inside) {
                if (!A_inside) {
                    result.vertices.push_back(P);
                }
                result.vertices.push_back(B);
            }
            else if (A_inside) {
                result.vertices.push_back(P);
            }
        }

        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell

};


// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);

    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();
  
   
    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    
    // Commented this code during lab 3

    /*
    for (int i = 0; i < n; i++) {
        ot->vor.weights[i] = x[i];
    }
    ot->vor.compute();

    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)
    
    double target_mass = 1.0 / n;
    
    for (int i = 0; i < n; i++) {
        double current_area = ot->vor.cells[i].area();
        double current_dist_integral = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        
        g[i] = -(target_mass - current_area);
        fx += -(current_dist_integral - x[i]*current_area + target_mass*x[i]);
    }
    */

    lbfgsfloatval_t fx = 0.0;

    int num_fluid = ot->vor.points.size();
    bool has_air = (n > num_fluid);
    double lam;

    if (has_air) {
        lam = target_fluid_area / num_fluid;
    } else {
        lam = 1.0 / n;
    }

    double sum_area = 0.0;

    for (int i = 0; i < num_fluid; i++) {
        double cell_area = ot->vor.cells[i].area();
        double dist_int  = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        g[i] = cell_area - lam;
        fx  += -(dist_int - x[i]*cell_area + lam*x[i]);
        sum_area += cell_area;
    }

    if (has_air) {
        g[num_fluid] = target_fluid_area - sum_area;
        fx += -(x[num_fluid] * (sum_area-target_fluid_area));
    }

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


// Lab 2
void OptimalTransport::optimize() {

    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);

    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer
    int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress, (void*)this, &param);

    // copy the result back to the voronoi structure
    vor.weights = weights;

    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 1000) : N_particles(N_particles) {
        // Implemented during lab 3

        fluid_volume = 0.2;
        target_fluid_area = fluid_volume;
        particles.resize(N_particles);
        velocities.resize(N_particles, Vector(0, 0));

        Vector blob_c(0.5, 0.7);
        double blob_r = sqrt(fluid_volume / PI);
        int placed = 0;

        while (placed < N_particles) {
            double rx = (double)rand() / RAND_MAX;
            double ry = (double)rand() / RAND_MAX;
            Vector cand(blob_c[0] - blob_r + 2*blob_r*rx,
                        blob_c[1] - blob_r + 2*blob_r*ry);
            if ((cand - blob_c).norm2() <= blob_r*blob_r) {
                particles[placed] = cand;
                placed ++;
            }
        }

        ot.vor.points = particles;
        ot.vor.weights.resize(N_particles + 1, fluid_volume / (N_particles * PI));
        ot.vor.weights[N_particles] = 0.0;
    }

    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {

        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;

        // TODO Lab 3 : 
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position

        ot.vor.points = particles;
        target_fluid_area = fluid_volume;
        ot.optimize();

        for (int i = 0; i < N_particles; i++) {
            Vector force = m_i * g;

            if (ot.vor.cells[i].vertices.size() >= 3) {
                Vector ctr = ot.vor.cells[i].centroid();
                force = force + (1.0 / epsilon2) * (ctr - particles[i]);
            }

            velocities[i] = velocities[i] + (dt/m_i) * force;
            particles[i]  = particles[i] + dt * velocities[i];

            if (particles[i][0] < 0) {
                particles[i][0] = 0;
                velocities[i][0] *= -0.5;
            }
            if (particles[i][0] > 1) {
                particles[i][0] = 1;
                velocities[i][0] *= -0.5;
            }
            
            if (particles[i][1] < 0) {
                particles[i][1] = 0;
                velocities[i][1] = 0;
            }
            if (particles[i][1] > 1) {
                particles[i][1] = 1;
                velocities[i][1] = 0;
            }
        }
    }

    // just run the full simulation
    void run_simulation() {
        system("mkdir -p frames");
        double dt = 0.002;
        
        for (int relax = 0; relax < 5; relax++) {
            ot.vor.points = particles;
            target_fluid_area = fluid_volume;
            ot.optimize();
            for (int i = 0; i < N_particles; i++) {
                if (ot.vor.cells[i].vertices.size() >= 3) {
                    particles[i] = ot.vor.cells[i].centroid();
                }
            }
        }
        for (int i = 0; i < 1000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "frames/test", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied by the fluid
};








int main() {

    /*
    Polygon p;
    p.vertices.push_back(Vector(0.1, 0.2));
    p.vertices.push_back(Vector(0.6, 0.4));
    p.vertices.push_back(Vector(0.5, 0.7));
    p.vertices.push_back(Vector(0.2, 0.5));

    std::vector<Polygon> s;
    s.push_back(p);

    save_frame(s, "toto");
    save_svg(s, "toto.svg");
    */

    // Commented this code during lab 3

    /*
    VoronoiDiagram vor;
    int N = 100;

    for (int i = 0; i < N; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        vor.points.push_back(Vector(x, y));
        vor.weights.push_back(0);
    }

    OptimalTransport ot;
    ot.vor = vor;

    ot.vor.compute();
    save_frame(ot.vor.cells, "voronoi");
    save_svg(ot.vor.cells, "voronoi.svg", &ot.vor.points);

    ot.optimize();
    save_frame(ot.vor.cells, "ot_result");
    save_svg(ot.vor.cells, "ot_result.svg", &ot.vor.points);
    */

    Fluid sim(300);
    sim.run_simulation();
    return 0;
}

// Compiled using: g++ -O3 -std=c++11 -fopenmp -I. main.cpp lbfgs.c -o main

// Video generated using: ffmpeg -framerate 24 -i frames/test%d.png -c:v libx264 -pix_fmt yuv420p output.mp4