from tkinter import *


class alexnet_gui:
    def __init__(self):
        window = Tk()
        theGui = gui(window)
        window.geometry("500x200")  # You want the size of the app to be 500x500
        window.resizable(0, 0)  # Don't allow resizing in the x or y direction
        window.title('GUI for Alexnet')
        window.mainloop()
        self.var = theGui.var


class gui:
    def __init__(self, master):
        self.master = master

        self.var = []

        label = Label(self.master, text='Set your parameters of alexnet')
        # label.config
        label.pack()
        frame1 = Frame(self.master)
        frame1.pack()
        self.btn2 = IntVar()

        frame2 = Frame(self.master)
        frame2.pack()
        # Label(frame1, text='Repetitions of Conv Block', font=("", 12)).grid(row=3, column=0, rowspan=50)
        subframe1 = Frame(frame2)

        default_r1 = StringVar()
        default_r1.set('2')
        self.conv2_x = Entry(subframe1, textvariable=default_r1)
        self.conv2_x.grid(row=3, column=1, columnspan=20)

        default_r2 = StringVar()
        default_r2.set('2')
        self.conv3_x = Entry(subframe1, textvariable=default_r2)
        self.conv3_x.grid(row=4, column=1, columnspan=20)

        default_r3 = StringVar()
        default_r3.set('2')
        self.conv4_x = Entry(subframe1, textvariable=default_r3)
        self.conv4_x.grid(row=5, column=1, columnspan=20)

        default_r4 = StringVar()
        default_r4.set('2')
        self.conv5_x = Entry(subframe1, textvariable=default_r4)
        self.conv5_x.grid(row=6, column=1, columnspan=20)

        subframe1.grid(row=3, column=0)
        frame3 = Frame(self.master)
        frame3.pack()
        button = Button(frame3, text='Run network', width=10, font=("", 12))
        button.bind('<Button-1>', self.get_variable)
        button.pack()

        self.quitBtn = Button(frame3, text="Quit", command=self.master.quit, font=("", 12)).pack()
        # get parameters of resnet

    def get_variable(self, event):
        self.var.append(int(self.conv2_x.get()))
        self.var.append(int(self.conv3_x.get()))
        self.var.append(int(self.conv4_x.get()))
        self.var.append(int(self.conv5_x.get()))
        self.var.append(int(self.btn2.get()))

        print(self.var)


