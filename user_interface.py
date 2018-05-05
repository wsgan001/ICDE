from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog, tkMessageBox
import ttk
from tkMessageBox  import askquestion, showerror
import pymysql as mdb
import os
import re

class App(object):
	def __init__(self, window,ip,user,pswd,db):
		self.ip=ip
		self.user=user
		self.pswd=pswd
		self.db=db

		window.wm_title("ICDE")
		window.geometry('550x250')

		#set the current row to zero
		self.current_row=0

		#tree view
		self.tree = ttk.Treeview(window)
		self.tree["columns"] = ("one","two", "three", "four", "five")
		self.tree.column("one",width=100)
		self.tree.column("two",width=100)
		self.tree.column("three",width=100)
		self.tree.column("four",width=100)
		self.tree.column("five",width=100)
		self.tree.column("#0",width=30,anchor='w')

		self.tree.grid(row=self.current_row, column=0, columnspan=6)
		self.tree.heading("#0",text="Id", anchor='w')
		self.tree.heading("one",text="Lastname")
		self.tree.heading("two",text="Firstname")
		self.tree.heading("three",text="Middlename")
		self.tree.heading("four",text="ID Type")
		self.tree.heading("five",text="Validity Date")
				
		#scrollbar
		self.scroll= Scrollbar(window)
		self.scroll.grid(row=self.current_row,column=7,rowspan=10)
		self.tree.configure(yscrollcommand=self.scroll.set)
		self.scroll.configure(command=self.tree.yview)
		self.current_row += 1

		#add button
		self.message = Label(text="", fg = "red").grid(row=self.current_row+1,column=3)
		self.update_btn=Button(window,text="Upload",width=12,command=self.upload)
		self.update_btn.grid(row=self.current_row,column=3)
		self.current_row += 1

		self.view_records()

	def view_records(self):
		records = self.tree.get_children()
		for element in records:
			self.tree.delete(element)
		conn = mdb.connect("localhost","root","Nutella4898","icdedb")
		cursor = conn.cursor()
		for i in range(0,6):
			cursor.execute("""SELECT * FROM cards ORDER BY lname""")
			
		for row in cursor:
			self.tree.insert('','end',text=row[0],values=(row[1],row[2],row[3],row[4],row[5]))

	def upload(self):
		self.upload_wind = Toplevel()
		self.upload_wind.title("Upload ID")
		Label(self.upload_wind, text = "File").grid(row=1, column=0)
		self.browse_btn=Button(self.upload_wind,text="Browse",width=6,command=self.load_file)
		self.browse_btn.grid(row=1,column=2)
		self.upload_wind.mainloop()

	def load_file(self):
		self.upload_wind.withdraw()
		self.filename = tkFileDialog.askopenfilename(initialdir = "tf_files/source/",title = "Browse file",filetypes = (("all files","*.*"),("jpg","*.jpg*"),("jpeg","*.jpeg*"),("png","*.png*")))
		filen = self.filename.split("/")
		path = filen[8]+"/"+filen[9]
		
		#run classifier
		print("\n==========Classifying ID Type...==========\n")
		os.system('python -m scripts.label_image --graph=tf_files/retrained_graph.pb  --image=tf_files/{0}'.format(path))
		
		#run extractor
		classified = open('tf_files/classified.txt','r').read()
		id_type = classified
		if id_type == "invalid":
			showerror('Error!', "Secondary IDs are considered as Invalid")
		else:
			print("\n========Extracting Name and Date...========\n")
			os.system('python scripts/data_extractor.py --image tf_files/{0} --type {1}'.format(path,id_type))
		self.view_records()
def main():
	window= Tk()
	start = App(window,"localhost","root","Nutella4898","icdedb")
	window.mainloop()

if __name__ == "__main__":
	main()