import predict
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import os


def upload_clicked():
    root.filename = filedialog.askopenfilename(
        initialdir=os.getcwd(), title="Odaberite sliku",
        filetypes=(("Sve slike", ("*.jpe", "*.jfif", "*.jpg", "*.jpeg", "*.png")), ("PNG", "*.png"), ("JPG", "*.jpg")))
    if root.filename != "":
        selected_img = ImageTk.PhotoImage(Image.open(root.filename).resize((256, 256)))
        row = list(root.grid_slaves(row=1))
        row[1].grid_forget()
        panel_bw = Label(root, image=selected_img)
        panel_bw.image = selected_img
        panel_bw.grid(row=1, column=0, padx=(25, 0), pady=25)


def colorize_clicked():
    global colorized_img
    model_path = os.path.join(os.getcwd(), 'model_cloud_3.h5')
    colorized_img = predict.predict_rgb_image(root.filename, model_path)
    colorized_photoimage = ImageTk.PhotoImage(colorized_img)
    panel_colorized = Label(root, image=colorized_photoimage)
    panel_colorized.image = colorized_photoimage
    panel_colorized.grid(row=1, column=2, padx=(0, 25), pady=25)


def save_clicked():
    current_colorized_image = colorized_img
    saveas = filedialog.asksaveasfile(mode="wb", initialdir=os.getcwd(), title="Spremite sliku",
                                      filetypes=(("PNG", "*.png"), ("JPG", "*.jpg")))
    current_colorized_image.save(saveas)


def main():
    root.configure(background="#1F1E44")
    root.resizable(False, False)
    root.title("Završni rad - aplikacija")
    icon_path = os.path.join(os.getcwd(), 'icon.ico')
    icon = PhotoImage(file=icon_path)
    root.tk.call('wm', 'iconphoto', root.w, icon)
    label_title = Label(root, text="Kolorizacija crno-bijelih slika", background="#1F1E44", foreground="white")
    label_title.config(font=("Tahoma", 19))
    button_upload = Button(root, text="Odaberi sliku", padx=50, command=upload_clicked)
    button_colorize = Button(root, text="Koloriziraj", padx=50, pady=5, command=colorize_clicked)
    button_save = Button(root, text="Spremi", padx=50, command=save_clicked)
    bw_frame = LabelFrame(root, background="#1c1a3c", height=256, width=256, text="Crno-bijela slika", foreground="white")
    bw_frame.grid(row=1, column=0, padx=(25, 0), pady=25)
    colorized_frame = LabelFrame(root, background="#1c1a3c", height=256, width=256, text="Kolorizirana slika", foreground="white")
    colorized_frame.grid(row=1, column=2, padx=(0, 25), pady=25)
    label_title.grid(row=0, column=1, pady=(20, 0))
    button_upload.grid(row=2, column=0, padx=(25, 0), pady=15)
    button_colorize.grid(row=1, column=1, pady=15)
    button_save.grid(row=2, column=2, padx=(0, 25), pady=15)
    ttk.Separator(root, orient=HORIZONTAL).grid(row=0, columnspan=3, sticky='ew', pady=(60, 0))
    root.mainloop()


if __name__ == '__main__':
    root = Tk()
    main()
