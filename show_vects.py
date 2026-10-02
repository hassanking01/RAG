import arcade
from sklearn.decomposition import PCA
import numpy
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path


class Circle(arcade.Sprite):
    def __init__(self, x, y, z, radius, color=arcade.color.RED):
        diameter = int(z * radius) * 2 if z * radius  > 1 else 1
        circle_texture = arcade.make_circle_texture(diameter, color)
        super().__init__(circle_texture)
        self.z = z
        self.center_x = int(x)
        self.center_y = int(y)
        self.radius = int(radius)
    def update(self, delta_time = 1 / 60, *args, **kwargs):
        radius = kwargs["radius"]
        diameter = int(self.z * radius) * 2 if self.z * radius  > 1 else 1
        circle_texture = arcade.make_circle_texture(diameter, arcade.color.RED)
        self.texture = circle_texture


class Screen(arcade.Window):
    def __init__(self):
        super().__init__(fullscreen=True)
        self.pca = PCA(n_components=3)
        self.embeddings = numpy.load("/home/hahchtar/Desktop/student/RAG/data/processed/semantic/embeddings.npy")
        self.embeddings = self.pca.fit_transform(self.embeddings)
        self.embeddings = self.embeddings * 1000
        self.camera = arcade.Camera2D()
        self.cx, self.cy = self.width // 2, self.height // 2
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.query = None
        self.is_query = False
        self.q_path = "/home/hahchtar/Desktop/student/RAG/data/input/public/AnsweredQuestions/dataset_code_public.json"
        self.questions = [data["question"]  for data in json.loads(Path(self.q_path).read_text())["rag_questions"]]
        self.vectors = arcade.SpriteList()
        self.r_init = 0.01
        self.quey_embidding = None
        for vector in self.embeddings:
                x, y , z = vector
                x += self.width // 2
                y += self.height // 2
                self.vectors.append(Circle(x, y, z , 0.01))
        self.index = 0
    # def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
    #     if int(self.r_init * 100) + scroll_y <= 0:
    #         return
    #     self.r_init += scroll_y / 100
    #     self.vectors.update(radius=self.r_init)
        

    def on_draw(self):
        self.clear()
        with self.camera.activate():
            self.vectors.draw()
        if self.is_query:
            self.on_query()
    # def on_text(self, text):
    #     self.user_input += text

    def on_query(self):
        r, g, b, _ = arcade.color.GRAY_BLUE
        rect = arcade.XYWH(self.width // 2, self.height // 2, self.width * 0.7, self.height * 0.7)
        arcade.draw.draw_rect_filled(rect, (r, g, b, 150))
        arcade.Text(
            "QUERY",
            self.width // 2,
            self.height // 2 + 300,
            arcade.color.RED,
            font_size=40,
            anchor_x="center"
        ).draw()
        y = self.height // 2 + 100
        for i in range(self.index - int(5 / 2), 5):
            arcade.Text(
                f"{self.questions[i % len(self.questions)]}",
                x=self.width // 2,
                y=y,
                color=arcade.color.BLACK if not i == self.index else arcade.color.GREEN_YELLOW,
                font_size=15 if not i == self.index != self.index else 20,
                anchor_x="center"
            ).draw()
            y -= 50
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.cx -= dx
        self.cy -= dy
        self.camera.position = (self.cx, self.cy)
    def get_query(self):
        flag = self.quey_embidding is not None
        self.quey_embidding = self.model.encode(self.questions[self.index])
        self.quey_embidding =  self.pca.transform(self.quey_embidding.reshape(1, -1)) * 1000
        if flag:
            self.vectors.pop()
        x, y, z = self.quey_embidding[0]
        self.vectors.append(Circle(x, y, z, 0.1, arcade.color.YELLOW))
    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.ESCAPE:
            self.is_query = not self.is_query
        if symbol == arcade.key.F:
            self.set_fullscreen(not self.fullscreen)
        if self.is_query:
            if symbol == arcade.key.DOWN:
                self.index -= 1
            if symbol == arcade.key.UP:
                self.index += 1
            if symbol == arcade.key.ENTER:
                self.get_query()
                self.is_query = False
if __name__ == "__main__":
    screen = Screen()
    screen.run()