#include "gpu.cuh"
#include "main.h"

using namespace std;

// size of the window (1024x1024)
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define LOCATION_OFFSET BUFFER_OFFSET(0)
#define COLOR_OFFSET BUFFER_OFFSET(16)

// global variables that will store handles to the data we
// intend to share between OpenGL and CUDA calculated data.
// handle for OpenGL side:
unsigned int vbo; // VBO for storing positions.
int n_vertices;
float delta;

// handle for CUDA side:
cudaGraphicsResource *resource;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

Vertex *devPtr;
float dt = 0;

// // Entry point for CUDA Kernel execution
// extern "C" void runCuda(cudaGraphicsResource** resource, Vertex* devPtr, int
// dim, float dt); extern "C" void unregRes(cudaGraphicsResource** res); extern
// "C" void chooseDev(int ARGC, const char **ARGV); extern "C" void
// regBuffer(cudaGraphicsResource** res, unsigned int& vbo);

void display(void) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);

  glColor3f(0.0, 1.0, 0.0);

  // render from the vbo
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(4, GL_FLOAT, sizeof(Vertex), LOCATION_OFFSET);

  glColorPointer(4, GL_FLOAT, sizeof(Vertex), COLOR_OFFSET);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, n_vertices);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glutSwapBuffers();
}

void idle(void) {
  dt += 0.01f;
  runCuda(&resource, devPtr, n_vertices, delta);
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1 << button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }

  mouse_old_x = x;
  mouse_old_y = y;
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouse_old_x);
  dy = (float)(y - mouse_old_y);

  if (mouse_buttons & 1) {
    rotate_x += dy * 0.2f;
    rotate_y += dx * 0.2f;
  } else if (mouse_buttons & 4) {
    translate_z += dy * 0.01f;
  }

  mouse_old_x = x;
  mouse_old_y = y;
}

void keys(unsigned char key, int x, int y) {
  static int toonf = 0;
  switch (key) {
  case 27:
    // clean up OpenGL and CUDA
    glDeleteBuffers(1, &vbo);
    // unregRes( &resource );
    exit(0);
    break;
  }
}

// Setting up the GL viewport and coordinate system
void reshape(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  gluPerspective(45, w * 1.0 / h * 1.0, 0.01, 10);
  // glTranslatef (0,0,-5);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

// glew (extension loading), OpenGL state and CUDA - OpenGL interoperability
// initialization
void initGL() {
  GLenum err = glewInit();
  if (GLEW_OK != err)
    return;
  cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << endl;
}

void initCUDA(int argc, const char **argv) {
  chooseDev(argc, argv);
  // creating a vertex buffer object in OpenGL and storing the handle in our
  // global variable GLuint vbo
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  // CARGAR DE UN ARCHIVO

  char **filename = (char **)malloc(sizeof(char *));

  if (!getCmdLineArgumentString(argc, argv, "file=", filename)) {
    cout << "Please specify an input file with the option --file." << endl;
    exit(EXIT_FAILURE);
  }

  ifstream input;
  input.open(*filename);

  if (!input) {
    cout << "Problem opening file." << endl;
    exit(EXIT_FAILURE);
  }

  input >> delta;
  input >> n_vertices;

  Vertex *v = new Vertex[n_vertices];

  float mass, position_x, position_y, position_z, speed_x, speed_y, speed_z;
  for (int i = 0; i < n_vertices; i++) {
    input >> mass >> position_x >> position_y >> position_z >> speed_x >>
        speed_y >> speed_z;

    v[i].mass = mass;

    v[i].position.x = position_x;
    v[i].position.y = position_y;
    v[i].position.z = position_z;
    v[i].position.w = 1.0f;

    v[i].speed.x = speed_x;
    v[i].speed.y = speed_y;
    v[i].speed.z = speed_z;

    v[i].acceleration.x = 0.0f;
    v[i].acceleration.y = 0.0f;
    v[i].acceleration.z = 0.0f;

    float cr = (float)(rand() % 502) + 10.0f;
    float cg = (float)(rand() % 502) + 10.0f;
    float cb = (float)(rand() % 502) + 10.0f;
    v[i].color.x = cr / (float)512;
    v[i].color.y = cg / (float)512;
    v[i].color.z = cb / (float)512;
    v[i].color.w = 1.0f;
  }

  input.close();

  glBufferData(GL_ARRAY_BUFFER, n_vertices * sizeof(Vertex), v,
               GL_DYNAMIC_DRAW);
  delete[] v;
  regBuffer(&resource, vbo);
  runCuda(&resource, devPtr, n_vertices, delta);
}

int main(int argc, const char **argv) {
  srand((unsigned int)time(NULL));
  // OpenGL configuration and GLUT calls  initialization
  // these need to be made before the other OpenGL
  // calls, else we get a seg fault
  glutInit(&argc, (char **)argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("1048576 points");
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keys);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  initGL();
  initCUDA(argc, argv);
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutMainLoop();

  return 0;
}
