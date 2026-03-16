-- ***************************************************************************
--                MNIST 2 Layer Neural Network
--
--           Copyright (C) 2026 By Ulrik Hørlyk Hjort
--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- ***************************************************************************

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Sequential_IO;
with Ada.Numerics.Float_Random;
with Ada.Numerics.Elementary_Functions;
with Interfaces; use Interfaces;

procedure MNIST_Network is
   
   -- Network architecture
   Input_Size : constant := 784;  -- 28x28 pixels
   Hidden_Size : constant := 128;
   Output_Size : constant := 10;  -- 10 digits (0-9)
   
   type Float_Array is array (Integer range <>) of Float;
   type Matrix is array (Integer range <>, Integer range <>) of Float;
   
   -- Network weights and biases
   W1 : Matrix (1 .. Hidden_Size, 1 .. Input_Size);
   B1 : Float_Array (1 .. Hidden_Size);
   W2 : Matrix (1 .. Output_Size, 1 .. Hidden_Size);
   B2 : Float_Array (1 .. Output_Size);
   
   -- Activations
   Hidden : Float_Array (1 .. Hidden_Size);
   Output : Float_Array (1 .. Output_Size);
   
   -- Training data
   type Image_Array is array (1 .. Input_Size) of Float;
   type Label is range 0 .. 9;
   
   -- Random number generator
   Gen : Ada.Numerics.Float_Random.Generator;
   
   -- Binary file IO - use Interfaces for proper byte type
   type Unsigned_8 is mod 2**8;
   package Byte_IO is new Ada.Sequential_IO (Unsigned_8);
   
   function Sigmoid (X : Float) return Float is
      use Ada.Numerics.Elementary_Functions;
   begin
      return 1.0 / (1.0 + Exp (-X));
   end Sigmoid;
   
   function ReLU (X : Float) return Float is
   begin
      return Float'Max (0.0, X);
   end ReLU;
   
   function ReLU_Derivative (X : Float) return Float is
   begin
      if X > 0.0 then
         return 1.0;
      else
         return 0.0;
      end if;
   end ReLU_Derivative;
   
   procedure Initialize_Weights is
      use Ada.Numerics.Float_Random;
   begin
      Reset (Gen);
      
      -- Initialize W1 with small random values
      for I in W1'Range (1) loop
         for J in W1'Range (2) loop
            W1 (I, J) := (Random (Gen) - 0.5) * 0.1;
         end loop;
         B1 (I) := 0.0;
      end loop;
      
      -- Initialize W2 with small random values
      for I in W2'Range (1) loop
         for J in W2'Range (2) loop
            W2 (I, J) := (Random (Gen) - 0.5) * 0.1;
         end loop;
         B2 (I) := 0.0;
      end loop;
   end Initialize_Weights;
   
   procedure Forward_Pass (Input_Data : Image_Array) is
   begin
      -- Hidden layer with ReLU activation
      for I in Hidden'Range loop
         Hidden (I) := B1 (I);
         for J in Input_Data'Range loop
            Hidden (I) := Hidden (I) + W1 (I, J) * Input_Data (J);
         end loop;
         Hidden (I) := ReLU (Hidden (I));
      end loop;
      
      -- Output layer with softmax (we'll use raw scores for now)
      for I in Output'Range loop
         Output (I) := B2 (I);
         for J in Hidden'Range loop
            Output (I) := Output (I) + W2 (I, J) * Hidden (J);
         end loop;
      end loop;
      
      -- Softmax normalization
      declare
         use Ada.Numerics.Elementary_Functions;
         Max_Val : Float := Output (Output'First);
         Sum : Float := 0.0;
      begin
         -- Find max for numerical stability
         for I in Output'Range loop
            if Output (I) > Max_Val then
               Max_Val := Output (I);
            end if;
         end loop;
         
         -- Compute exp and sum
         for I in Output'Range loop
            Output (I) := Exp (Output (I) - Max_Val);
            Sum := Sum + Output (I);
         end loop;
         
         -- Normalize
         for I in Output'Range loop
            Output (I) := Output (I) / Sum;
         end loop;
      end;
   end Forward_Pass;
   
   function Predict (Input_Data : Image_Array) return Label is
      Max_Idx : Integer := Output'First;
      Max_Val : Float := Output (Output'First);
   begin
      Forward_Pass (Input_Data);
      
      for I in Output'Range loop
         if Output (I) > Max_Val then
            Max_Val := Output (I);
            Max_Idx := I;
         end if;
      end loop;
      
      return Label (Max_Idx - 1);  -- Convert to 0-9
   end Predict;
   
   function Read_Int32_BE (File : Byte_IO.File_Type) return Natural is
      B1, B2, B3, B4 : Unsigned_8;
      Result : Long_Integer;
   begin
      Byte_IO.Read (File, B1);
      Byte_IO.Read (File, B2);
      Byte_IO.Read (File, B3);
      Byte_IO.Read (File, B4);
      
      -- Big endian: most significant byte first
      Result := Long_Integer (B1) * 16777216 +  -- 256^3
                Long_Integer (B2) * 65536 +      -- 256^2
                Long_Integer (B3) * 256 + 
                Long_Integer (B4);
      
      return Natural (Result);
   end Read_Int32_BE;
   
   procedure Read_MNIST_Images (Filename : String; 
                                 Num_Images : out Natural) is
      File : Byte_IO.File_Type;
      Magic, Count, Rows, Cols : Natural;
   begin
      Byte_IO.Open (File, Byte_IO.In_File, Filename);
      
      -- Read header (16 bytes total)
      Magic := Read_Int32_BE (File);  -- Magic number
      Count := Read_Int32_BE (File);  -- Number of images
      Rows := Read_Int32_BE (File);   -- Number of rows
      Cols := Read_Int32_BE (File);   -- Number of columns
      
      Num_Images := Count;
      Put_Line ("Magic: " & Natural'Image (Magic));
      Put_Line ("Images: " & Natural'Image (Count));
      Put_Line ("Rows: " & Natural'Image (Rows));
      Put_Line ("Cols: " & Natural'Image (Cols));
      
      Byte_IO.Close (File);
   end Read_MNIST_Images;
   
   procedure Load_Image (File : Byte_IO.File_Type; 
                         Image : out Image_Array) is
      B : Unsigned_8;
   begin
      for I in Image'Range loop
         Byte_IO.Read (File, B);
         Image (I) := Float (B) / 255.0;  -- Normalize to 0..1
      end loop;
   end Load_Image;
   
   procedure Read_MNIST_Labels (Filename : String;
                                Num_Labels : out Natural) is
      File : Byte_IO.File_Type;
      Magic, Count : Natural;
   begin
      Byte_IO.Open (File, Byte_IO.In_File, Filename);
      
      -- Read header (8 bytes total)
      Magic := Read_Int32_BE (File);  -- Magic number
      Count := Read_Int32_BE (File);  -- Number of labels
      
      Num_Labels := Count;
      Put_Line ("Label Magic: " & Natural'Image (Magic));
      Put_Line ("Labels: " & Natural'Image (Count));
      
      Byte_IO.Close (File);
   end Read_MNIST_Labels;
   
   function Load_Label (File : Byte_IO.File_Type) return Label is
      B : Unsigned_8;
   begin
      Byte_IO.Read (File, B);
      return Label (B);
   end Load_Label;
   
   procedure Train_Network (Num_Epochs : Natural; 
                            Learning_Rate : Float;
                            Batch_Size : Natural) is
      Image_File : Byte_IO.File_Type;
      Label_File : Byte_IO.File_Type;
      Magic, Count, Rows, Cols : Natural;
      Current_Image : Image_Array;
      Current_Label : Label;
      Target : Float_Array (1 .. Output_Size);
      
      -- Gradients
      DW1 : Matrix (1 .. Hidden_Size, 1 .. Input_Size) := (others => (others => 0.0));
      DB1 : Float_Array (1 .. Hidden_Size) := (others => 0.0);
      DW2 : Matrix (1 .. Output_Size, 1 .. Hidden_Size) := (others => (others => 0.0));
      DB2 : Float_Array (1 .. Output_Size) := (others => 0.0);
      
      Hidden_Pre : Float_Array (1 .. Hidden_Size);
      Output_Error : Float_Array (1 .. Output_Size);
      Hidden_Error : Float_Array (1 .. Hidden_Size);
      
      Correct : Natural;
      Total_Loss : Float;
   begin
      for Epoch in 1 .. Num_Epochs loop
         -- Open files
         Byte_IO.Open (Image_File, Byte_IO.In_File, "./data/train-images-idx3-ubyte");
         Byte_IO.Open (Label_File, Byte_IO.In_File, "./data/train-labels-idx1-ubyte");
         
         -- Skip headers
         Magic := Read_Int32_BE (Image_File);
         Count := Read_Int32_BE (Image_File);
         Rows := Read_Int32_BE (Image_File);
         Cols := Read_Int32_BE (Image_File);
         
         Magic := Read_Int32_BE (Label_File);
         Count := Read_Int32_BE (Label_File);
         
         Correct := 0;
         Total_Loss := 0.0;
         
         -- Train on all images
         for I in 1 .. Count loop
            -- Load image and label
            Load_Image (Image_File, Current_Image);
            Current_Label := Load_Label (Label_File);
            
            -- Create one-hot target
            Target := (others => 0.0);
            Target (Integer (Current_Label) + 1) := 1.0;
            
            -- Forward pass
            --Forward_Pass (Current_Image);
            
            -- Compute loss (cross-entropy)
            for J in Output'Range loop
               if Output (J) > 0.0 then
                  Total_Loss := Total_Loss - Target (J) * 
                                Ada.Numerics.Elementary_Functions.Log (Output (J));
               end if;
            end loop;
            
            -- Check prediction
            declare
               Pred : constant Label := Predict (Current_Image);
            begin
               if Pred = Current_Label then
                  Correct := Correct + 1;
               end if;
            end;
            
            -- Backpropagation
            -- Output layer error
            for J in Output'Range loop
               Output_Error (J) := Output (J) - Target (J);
            end loop;
            
            -- Hidden layer error
            for J in Hidden'Range loop
               Hidden_Error (J) := 0.0;
               for K in Output'Range loop
                  Hidden_Error (J) := Hidden_Error (J) + 
                                      W2 (K, J) * Output_Error (K);
               end loop;
               Hidden_Error (J) := Hidden_Error (J) * ReLU_Derivative (Hidden (J));
            end loop;
            
            -- Accumulate gradients
            for J in W2'Range (1) loop
               for K in W2'Range (2) loop
                  DW2 (J, K) := DW2 (J, K) + Output_Error (J) * Hidden (K);
               end loop;
               DB2 (J) := DB2 (J) + Output_Error (J);
            end loop;
            
            for J in W1'Range (1) loop
               for K in W1'Range (2) loop
                  DW1 (J, K) := DW1 (J, K) + Hidden_Error (J) * Current_Image (K);
               end loop;
               DB1 (J) := DB1 (J) + Hidden_Error (J);
            end loop;
            
            -- Update weights every batch_size samples
            if I mod Batch_Size = 0 then
               declare
                  Scale : constant Float := Learning_Rate / Float (Batch_Size);
               begin
                  for J in W2'Range (1) loop
                     for K in W2'Range (2) loop
                        W2 (J, K) := W2 (J, K) - Scale * DW2 (J, K);
                     end loop;
                     B2 (J) := B2 (J) - Scale * DB2 (J);
                  end loop;
                  
                  for J in W1'Range (1) loop
                     for K in W1'Range (2) loop
                        W1 (J, K) := W1 (J, K) - Scale * DW1 (J, K);
                     end loop;
                     B1 (J) := B1 (J) - Scale * DB1 (J);
                  end loop;
                  
                  -- Reset gradients
                  DW1 := (others => (others => 0.0));
                  DB1 := (others => 0.0);
                  DW2 := (others => (others => 0.0));
                  DB2 := (others => 0.0);
               end;
            end if;
         end loop;
         
         Byte_IO.Close (Image_File);
         Byte_IO.Close (Label_File);
         
         -- Print epoch statistics
         Put_Line ("Epoch " & Natural'Image (Epoch) & 
                   " - Train Accuracy: " & 
                   Float'Image (Float (Correct) * 100.0 / Float (Count)) & "%" &
                   " Loss: " & Float'Image (Total_Loss / Float (Count)));
      end loop;
   end Train_Network;
   
   procedure Test_Network is
      Image_File : Byte_IO.File_Type;
      Label_File : Byte_IO.File_Type;
      Magic, Count, Rows, Cols : Natural;
      Current_Image : Image_Array;
      Current_Label : Label;
      Correct : Natural := 0;
      Pred : Label;
      
      -- Confusion matrix for detailed analysis
      Confusion : array (0 .. 9, 0 .. 9) of Natural := (others => (others => 0));
   begin
      Put_Line ("Testing on test set...");
      
      -- Open test files
      Byte_IO.Open (Image_File, Byte_IO.In_File, "./data/t10k-images-idx3-ubyte");
      Byte_IO.Open (Label_File, Byte_IO.In_File, "./data/t10k-labels-idx1-ubyte");
      
      -- Skip headers
      Magic := Read_Int32_BE (Image_File);
      Count := Read_Int32_BE (Image_File);
      Rows := Read_Int32_BE (Image_File);
      Cols := Read_Int32_BE (Image_File);
      
      Magic := Read_Int32_BE (Label_File);
      Count := Read_Int32_BE (Label_File);
      
      Put_Line ("Test images: " & Natural'Image (Count));
      
      -- Test on all images
      for I in 1 .. Count loop
         Load_Image (Image_File, Current_Image);
         Current_Label := Load_Label (Label_File);
         
         Pred := Predict (Current_Image);
         
         -- Update confusion matrix
         Confusion (Integer (Current_Label), Integer (Pred)) := 
            Confusion (Integer (Current_Label), Integer (Pred)) + 1;
         
         if Pred = Current_Label then
            Correct := Correct + 1;
         end if;
      end loop;
      
      Byte_IO.Close (Image_File);
      Byte_IO.Close (Label_File);
      
      -- Print results
      New_Line;
      Put_Line ("========================================");
      Put_Line ("TEST SET RESULTS");
      Put_Line ("========================================");
      Put_Line ("Total test images: " & Natural'Image (Count));
      Put_Line ("Correct predictions: " & Natural'Image (Correct));
      Put_Line ("Test Accuracy: " & 
                Float'Image (Float (Correct) * 100.0 / Float (Count)) & "%");
      New_Line;
      
      -- Print per-digit accuracy
      Put_Line ("Per-digit accuracy:");
      for Digit in 0 .. 9 loop
         declare
            Total_For_Digit : Natural := 0;
            Correct_For_Digit : Natural := Confusion (Digit, Digit);
         begin
            for Pred_Digit in 0 .. 9 loop
               Total_For_Digit := Total_For_Digit + Confusion (Digit, Pred_Digit);
            end loop;
            
            if Total_For_Digit > 0 then
               Put_Line ("  Digit " & Integer'Image (Digit) & ": " &
                        Float'Image (Float (Correct_For_Digit) * 100.0 / 
                                    Float (Total_For_Digit)) & "%");
            end if;
         end;
      end loop;
   end Test_Network;
   
   Num_Images, Num_Labels : Natural;
   
begin
   Put_Line ("MNIST Neural Network in Ada");
   Put_Line ("============================");
   New_Line;
   
   -- Read dataset info
   Put_Line ("Reading training images...");
   Read_MNIST_Images ("./data/train-images-idx3-ubyte", Num_Images);
   New_Line;
   
   Put_Line ("Reading training labels...");
   Read_MNIST_Labels ("./data/train-labels-idx1-ubyte", Num_Labels);
   New_Line;
   
   -- Initialize network
   Put_Line ("Initializing neural network...");
   Put_Line ("Architecture: " & Integer'Image (Input_Size) & " -> " &
             Integer'Image (Hidden_Size) & " -> " &
             Integer'Image (Output_Size));
   Initialize_Weights;
   New_Line;
   
   Put_Line ("Starting training...");
   Put_Line ("Learning rate: 0.01, Batch size: 32, Epochs: 5");
   New_Line;
   
   Train_Network (Num_Epochs => 5, 
                  Learning_Rate => 0.01,
                  Batch_Size => 32);
   
   New_Line;
   Put_Line ("Training complete!");
   New_Line;
   
   -- Test on test set
   Test_Network;
   
end MNIST_Network;