var bytes = new byte[60];
new System.Security.Cryptography.RNGCryptoServiceProvider().GetBytes(bytes);
Console.WriteLine(Convert.ToBase64String(bytes));
