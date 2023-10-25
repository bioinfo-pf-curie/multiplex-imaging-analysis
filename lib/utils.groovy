
class utils {
    public static String getFileID(f, delim) {
        return f.getName().toString().split(delim).head()
    }
}